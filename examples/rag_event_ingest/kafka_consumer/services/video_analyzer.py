# services/video_analyzer.py
"""Video analysis service using VSS 2.4 API."""

import json
import re
import logging
from pathlib import Path
from typing import Optional, Tuple
import requests

from config import (
    API_VSS_FILES,
    API_VSS_SUMMARIZE,
    CONTENT_TYPE_MAP,
    DEFAULT_CONTENT_TYPE,
    VSS_DEFAULT_PROMPT,
    VSS_SYSTEM_PROMPT,
    VSS_CAPTION_SUMMARIZATION_PROMPT,
    VSS_SUMMARY_AGGREGATION_PROMPT,
    VSS_CHUNK_DURATION,
    VSS_CHUNK_OVERLAP,
    VSS_NUM_FRAMES_PER_CHUNK,
    VSS_MAX_TOKENS,
    VSS_MODEL,
    VSS_UPLOAD_TIMEOUT,
    VSS_STREAM_ENABLED,
    # API fields
    FIELD_ID,
    FIELD_FILE,
    FIELD_MODEL,
    FIELD_PROMPT,
    FIELD_SYSTEM_PROMPT,
    FIELD_MAX_TOKENS,
    FIELD_CHUNK_DURATION,
    FIELD_CHUNK_OVERLAP_DURATION,
    FIELD_NUM_FRAMES_PER_CHUNK,
    FIELD_CAPTION_SUMMARIZATION_PROMPT,
    FIELD_SUMMARY_AGGREGATION_PROMPT,
    FIELD_PURPOSE,
    FIELD_MEDIA_TYPE,
    FIELD_STREAM,
    VALUE_VISION,
    VALUE_VIDEO,
    RESP_CONTENT,
    RESP_RESPONSE,
    RESP_TEXT,
    RESP_CHOICES,
    RESP_MESSAGE,
    RESP_DELTA,
    RESP_DATA_PREFIX,
)

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Analyzes videos using VSS 2.4 /files and /summarize APIs."""
    
    def __init__(self, base_url: str, timeout: int = 1800):
        """Initialize video analyzer.
        
        Args:
            base_url: VSS server URL
            timeout: Timeout for summarization (default 30 min for long videos)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        logger.info(f"VideoAnalyzer initialized (VSS 2.4): {self.base_url}")
    
    def upload_video(self, video_data: bytes, filename: str) -> Tuple[bool, Optional[str]]:
        """Upload video to VSS using /files API.
        
        Args:
            video_data: Video file bytes
            filename: Original filename
            
        Returns:
            Tuple of (success, file_id)
        """
        file_name_sanitized = self._sanitize_filename(filename)
        content_type = self._get_content_type(file_name_sanitized)
        
        logger.info(f"Uploading video to VSS: {file_name_sanitized} ({len(video_data)} bytes)")
        
        try:
            response = requests.post(
                f'{self.base_url}{API_VSS_FILES}',
                files={FIELD_FILE: (file_name_sanitized, video_data, content_type)},
                data={FIELD_PURPOSE: VALUE_VISION, FIELD_MEDIA_TYPE: VALUE_VIDEO},
                timeout=VSS_UPLOAD_TIMEOUT
            )
        except requests.RequestException as e:
            logger.error(f"Error uploading video to VSS: {e}")
            return False, None
        
        if response.status_code not in [200, 201]:
            logger.error(f"Upload failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False, None
        
        result = response.json()
        file_id = result.get(FIELD_ID)
        
        if not file_id:
            logger.error(f"No file ID in response: {result}")
            return False, None
        
        logger.info(f"✓ Video uploaded: {file_name_sanitized} (id: {file_id})")
        return True, file_id
    
    def get_video_description(self, file_id: str) -> Optional[str]:
        """Get video description using /summarize API.
        
        Uses streaming mode for real-time progress updates if enabled.
        
        Args:
            file_id: File ID from upload
            
        Returns:
            Video summary text, or None if failed
        """
        if VSS_STREAM_ENABLED:
            return self._get_description_streaming(file_id)
        return self._get_description_blocking(file_id)
    
    def _build_summarize_payload(self, file_id: str, stream: bool = False) -> dict:
        """Build the payload for summarize request."""
        payload = {
            FIELD_ID: file_id,
            FIELD_MODEL: VSS_MODEL,
            FIELD_PROMPT: VSS_DEFAULT_PROMPT,
            FIELD_SYSTEM_PROMPT: VSS_SYSTEM_PROMPT,
            FIELD_MAX_TOKENS: VSS_MAX_TOKENS,
            FIELD_STREAM: stream,
        }
        
        # Add chunking for long videos
        if VSS_CHUNK_DURATION > 0:
            payload.update({
                FIELD_CHUNK_DURATION: VSS_CHUNK_DURATION,
                FIELD_CHUNK_OVERLAP_DURATION: VSS_CHUNK_OVERLAP,
                FIELD_NUM_FRAMES_PER_CHUNK: VSS_NUM_FRAMES_PER_CHUNK,
                FIELD_CAPTION_SUMMARIZATION_PROMPT: VSS_CAPTION_SUMMARIZATION_PROMPT,
                FIELD_SUMMARY_AGGREGATION_PROMPT: VSS_SUMMARY_AGGREGATION_PROMPT,
            })
        
        return payload
    
    def _get_description_blocking(self, file_id: str) -> Optional[str]:
        """Get video description using blocking mode."""
        logger.info(f"Requesting summarization (blocking) for file: {file_id}")
        
        payload = self._build_summarize_payload(file_id, stream=False)
        
        if VSS_CHUNK_DURATION > 0:
            logger.info(f"Using chunked processing: {VSS_CHUNK_DURATION}s chunks, "
                       f"{VSS_NUM_FRAMES_PER_CHUNK} frames/chunk")
        
        try:
            response = requests.post(
                f'{self.base_url}{API_VSS_SUMMARIZE}',
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout
            )
        except requests.RequestException as e:
            logger.error(f"Error requesting video summary: {e}")
            return None
        
        if response.status_code != 200:
            logger.error(f"Summarize failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
        
        result = response.json()
        content = self._extract_content(result)
        
        if not content:
            logger.warning("No content in summarize response")
            return None
        
        logger.info(f"✓ Got video summary ({len(content)} chars)")
        return content
    
    def _get_description_streaming(self, file_id: str) -> Optional[str]:
        """Get video description using streaming mode (Server-Sent Events)."""
        logger.info(f"Requesting summarization (streaming) for file: {file_id}")
        
        payload = self._build_summarize_payload(file_id, stream=True)
        
        if VSS_CHUNK_DURATION > 0:
            logger.info(f"Using chunked processing: {VSS_CHUNK_DURATION}s chunks, "
                       f"{VSS_NUM_FRAMES_PER_CHUNK} frames/chunk")
        
        try:
            response = requests.post(
                f'{self.base_url}{API_VSS_SUMMARIZE}',
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout,
                stream=True  # Enable streaming response
            )
        except requests.RequestException as e:
            logger.error(f"Error requesting video summary (streaming): {e}")
            return None
        
        if response.status_code != 200:
            logger.error(f"Summarize failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
        
        # Collect streaming chunks
        content = self._collect_stream_content(response)
        
        if not content:
            logger.warning("No content in streaming response")
            return None
        
        logger.info(f"✓ Got video summary via streaming ({len(content)} chars)")
        return content
    
    def _collect_stream_content(self, response) -> Optional[str]:
        """Collect content from streaming response (Server-Sent Events)."""
        chunks = []
        chunk_count = 0
        
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            
            # SSE format: "data: {...json...}"
            if line.startswith(RESP_DATA_PREFIX):
                json_str = line[len(RESP_DATA_PREFIX):]
                
                # Handle [DONE] marker
                if json_str.strip() == '[DONE]':
                    logger.info("Stream completed")
                    break
                
                try:
                    data = json.loads(json_str)
                    content = self._extract_stream_chunk(data)
                    if content:
                        chunks.append(content)
                        chunk_count += 1
                        if chunk_count % 10 == 0:
                            logger.info(f"Received {chunk_count} chunks...")
                    elif chunk_count == 0:
                        # Debug: log first non-content chunk to understand format
                        logger.debug(f"SSE data (no content): {json_str[:200]}")
                except json.JSONDecodeError as e:
                    logger.debug(f"SSE JSON parse error: {e}, line: {json_str[:100]}")
                    continue
            else:
                # Non-data line - could be event type or comment
                if chunk_count == 0:
                    logger.debug(f"SSE non-data line: {line[:100]}")
        
        if not chunks:
            logger.warning("No streaming chunks collected, checking full response")
            return None
        
        logger.info(f"Collected {chunk_count} streaming chunks")
        return ''.join(chunks)
    
    def _extract_stream_chunk(self, data: dict) -> Optional[str]:
        """Extract content from a single streaming chunk."""
        choices = data.get(RESP_CHOICES, [])
        if choices:
            delta = choices[0].get(RESP_DELTA, {})
            content = delta.get(RESP_CONTENT)
            if content:
                return content
            
            message = choices[0].get(RESP_MESSAGE, {})
            content = message.get(RESP_CONTENT)
            if content:
                return content
        
        if RESP_CONTENT in data:
            return data[RESP_CONTENT]
        
        if RESP_RESPONSE in data:
            return data[RESP_RESPONSE]
        
        if RESP_TEXT in data:
            return data[RESP_TEXT]
        
        return None
    
    def _extract_content(self, result: dict) -> Optional[str]:
        """Extract text content from VSS response.
        
        returns OpenAI-compatible format:
        {
            "choices": [{"message": {"content": "..."}}]
        }
        """
        # Try OpenAI format first
        choices = result.get(RESP_CHOICES, [])
        if choices:
            message = choices[0].get(RESP_MESSAGE, {})
            content = message.get(RESP_CONTENT)
            if content:
                return content.strip()
        
        if RESP_CONTENT in result:
            return result[RESP_CONTENT].strip()
        
        if RESP_RESPONSE in result:
            return result[RESP_RESPONSE].strip()
        
        return None
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for VSS compatibility."""
        name = Path(filename).name
        file_name_sanitized = re.sub(r'[^A-Za-z0-9_.\-]', '_', name)
        file_name_sanitized = re.sub(r'_+', '_', file_name_sanitized)
        return file_name_sanitized
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type from filename."""
        ext = Path(filename).suffix.lower()
        return CONTENT_TYPE_MAP.get(ext, DEFAULT_CONTENT_TYPE)

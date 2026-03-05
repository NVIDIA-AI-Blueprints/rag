# services/video_analyzer.py
"""Video analysis service using LVS API + media storage.

Videos are uploaded to a media storage service so LVS can access them
via an internal URL.  LVS returns structured JSON with numeric
timestamps (seconds); this module post-processes the output to embed
human-readable [MM:SS] timestamps inline, improving semantic search
accuracy for time-range queries.
"""

import json
import logging
import tempfile
from pathlib import Path, PurePosixPath
from typing import Optional, List, Tuple
from urllib.parse import urlencode

import ffmpy
import requests

from config import (
    API_VSS_SUMMARIZE,
    API_VST_STORAGE_UPLOAD,
    TRANSCODE_CONTAINER,
    TRANSCODE_FFMPEG_OPTS,
    VSS_CHUNK_DURATION,
    VSS_CHUNK_OVERLAP,
    VSS_NUM_FRAMES_PER_CHUNK,
    VSS_MAX_TOKENS,
    VSS_MODEL,
    VSS_SCENARIO,
    VSS_EVENTS,
    VSS_OBJECTS_OF_INTEREST,
    FIELD_MODEL,
    FIELD_MAX_TOKENS,
    FIELD_CHUNK_DURATION,
    FIELD_CHUNK_OVERLAP_DURATION,
    FIELD_NUM_FRAMES_PER_CHUNK,
    FIELD_URL,
    FIELD_SCENARIO,
    FIELD_EVENTS,
    FIELD_OBJECTS_OF_INTEREST,
    RESP_CONTENT,
    RESP_RESPONSE,
    RESP_CHOICES,
    RESP_MESSAGE,
    RESP_EVENTS,
    RESP_START_TIME,
    RESP_END_TIME,
    RESP_TYPE,
    RESP_DESCRIPTION,
)

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Analyzes videos using LVS /summarize API.

    Videos are first uploaded to media storage, then LVS fetches
    them via an internal URL.  The raw LVS JSON is post-processed
    into readable text with inline [MM:SS] timestamps.
    """

    def __init__(self, base_url: str, storage_url: str = '', timeout: int = 1800):
        self.base_url = base_url.rstrip('/')
        self.storage_url = storage_url.rstrip('/') if storage_url else ''
        self.timeout = timeout
        logger.info(f"VideoAnalyzer initialized — LVS: {self.base_url}, storage: {self.storage_url}")

    # ------------------------------------------------------------------
    # Media Storage
    # ------------------------------------------------------------------

    def upload_video(self, video_data: bytes, filename: str) -> Optional[str]:
        """Upload video to media storage and return a download URL.

        If storage rejects the codec (HTTP 422), the video is automatically
        transcoded and the upload is retried once.

        Returns:
            Download URL for the uploaded file, or *None* on failure.
        """
        if not self.storage_url:
            logger.error("storage_url not configured")
            return None

        safe_name = PurePosixPath(filename).name
        resp = self._put_to_storage(video_data, safe_name)
        if resp is None:
            return None

        if resp.status_code == 422:
            logger.warning(f"Storage rejected format, transcoding: {resp.text}")
            video_data, safe_name = self._transcode(video_data, safe_name)
            resp = self._put_to_storage(video_data, safe_name)
            if resp is None or resp.status_code != 200:
                logger.error("Upload failed after transcode")
                return None

        if resp.status_code != 200:
            logger.error(f"Storage upload failed: {resp.status_code} — {resp.text}")
            return None

        file_id = resp.json().get('id')
        if not file_id:
            logger.error("Storage upload response missing 'id'")
            return None

        download_url = (
            f"{self.storage_url}{API_VST_STORAGE_UPLOAD}"
            f"?{urlencode({'id': file_id, 'filename': safe_name})}"
        )
        logger.info(f"Uploaded to storage: {safe_name} -> {file_id}")
        return download_url

    def _put_to_storage(self, video_data: bytes, filename: str) -> Optional[requests.Response]:
        """PUT raw bytes to media storage. Returns the response or None."""
        url = f"{self.storage_url}{API_VST_STORAGE_UPLOAD}/{filename}"
        try:
            return requests.put(
                url,
                data=video_data,
                headers={'Content-Type': 'application/octet-stream'},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            logger.error(f"Storage upload failed: {e}")
            return None

    def _transcode(self, video_data: bytes, filename: str) -> Tuple[bytes, str]:
        """Transcode video to a compatible codec via a temp file pair."""
        out_name = Path(filename).stem + TRANSCODE_CONTAINER
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / filename
            out_path = Path(tmp) / ('out_' + out_name)
            in_path.write_bytes(video_data)

            ff = ffmpy.FFmpeg(
                inputs={str(in_path): None},
                outputs={str(out_path): TRANSCODE_FFMPEG_OPTS},
            )
            logger.info(f"Transcoding: {ff.cmd}")
            ff.run()

            transcoded = out_path.read_bytes()
            logger.info(f"Transcoded {filename} -> {out_name} "
                        f"({len(video_data)} -> {len(transcoded)} bytes)")
            return transcoded, out_name

    # ------------------------------------------------------------------
    # LVS Summarization
    # ------------------------------------------------------------------

    def summarize(self, video_url: str) -> Optional[str]:
        """Summarize a video and return formatted text with MM:SS timestamps.

        Args:
            video_url: HTTP URL accessible by LVS (typically a VST download URL).

        Returns:
            Formatted text with [MM:SS] timestamps, or None on failure.
        """
        logger.info(f"Requesting LVS summarization: {video_url}")

        payload = {
            FIELD_URL: video_url,
            FIELD_MODEL: VSS_MODEL,
            FIELD_SCENARIO: VSS_SCENARIO,
            FIELD_EVENTS: VSS_EVENTS,
            FIELD_OBJECTS_OF_INTEREST: VSS_OBJECTS_OF_INTEREST,
            FIELD_MAX_TOKENS: VSS_MAX_TOKENS,
        }

        if VSS_CHUNK_DURATION > 0:
            payload.update({
                FIELD_CHUNK_DURATION: VSS_CHUNK_DURATION,
                FIELD_CHUNK_OVERLAP_DURATION: VSS_CHUNK_OVERLAP,
                FIELD_NUM_FRAMES_PER_CHUNK: VSS_NUM_FRAMES_PER_CHUNK,
            })

        raw = self._post(payload)
        if not raw:
            return None

        return self._format_events(raw)

    # ------------------------------------------------------------------
    # Timestamp formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _secs_to_mmss(seconds: float) -> str:
        total = int(seconds)
        return f"{total // 60:02d}:{total % 60:02d}"

    def _format_events(self, raw_content: str) -> str:
        """Convert LVS structured JSON into readable text with MM:SS timestamps.

        Input:
            {"events": [{"start_time": 960.03, "end_time": 1020.05,
                          "type": "touchdown", "description": "..."}]}

        Output:
            [16:00-17:00] touchdown: A Seahawks player scores...
        """
        try:
            data = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            logger.debug("LVS content is not JSON, returning as-is")
            return raw_content

        events = data.get(RESP_EVENTS, [])
        if not events:
            logger.debug("No events in LVS JSON, returning raw content")
            return raw_content

        lines: List[str] = []
        for ev in events:
            st = ev.get(RESP_START_TIME, 0)
            et = ev.get(RESP_END_TIME, st)
            etype = ev.get(RESP_TYPE, '')
            desc = ev.get(RESP_DESCRIPTION, '')

            ts_start = self._secs_to_mmss(st)
            if abs(et - st) < 1:
                ts = f"[{ts_start}]"
            else:
                ts = f"[{ts_start}-{self._secs_to_mmss(et)}]"

            prefix = f"{ts} {etype}:" if etype else ts
            lines.append(f"{prefix} {desc}")

        formatted = '\n'.join(lines)
        logger.info(f"Formatted {len(events)} events with MM:SS timestamps")
        return formatted

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _post(self, payload: dict) -> Optional[str]:
        try:
            response = requests.post(
                f'{self.base_url}{API_VSS_SUMMARIZE}',
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            logger.error(f"Error requesting video summary: {e}")
            return None

        if response.status_code != 200:
            logger.error(f"Summarize failed: {response.status_code} — {response.text}")
            return None

        content = self._extract_content(response.json())
        if not content:
            logger.warning("No content in summarize response")
            return None

        logger.info(f"Got video summary ({len(content)} chars)")
        return content

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _extract_content(self, result: dict) -> Optional[str]:
        """Extract text from OpenAI-compatible response."""
        choices = result.get(RESP_CHOICES, [])
        if choices:
            message = choices[0].get(RESP_MESSAGE, {})
            content = message.get(RESP_CONTENT)
            if content:
                return content.strip()

        for key in (RESP_CONTENT, RESP_RESPONSE):
            if key in result:
                return result[key].strip()

        return None

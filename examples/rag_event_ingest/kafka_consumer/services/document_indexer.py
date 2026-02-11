# services/document_indexer.py
"""Document indexing service for RAG pipeline."""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple
import requests

from config import (
    API_INGESTOR_DOCUMENTS,
    API_INGESTOR_COLLECTIONS,
    API_INGESTOR_COLLECTION,
    API_INGESTOR_STATUS,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_FINISHED,
    STATUS_FAILED,
    TIMEOUT_DEFAULT,
    TIMEOUT_MAX_TASK_WAIT,
    VIDEO_INDEX_TIMEOUT,
    COLLECTION_EMBEDDING_DIMENSION,
    COLLECTION_CHUNK_SIZE,
    COLLECTION_CHUNK_OVERLAP,
    COLLECTION_VIDEO_CHUNK_SIZE,
    COLLECTION_VIDEO_CHUNK_OVERLAP,
    CONTENT_TYPE_MAP,
    DEFAULT_CONTENT_TYPE,
    FIELD_COLLECTION_NAME,
    FIELD_BLOCKING,
    FIELD_SPLIT_OPTIONS,
    FIELD_CHUNK_SIZE,
    FIELD_CHUNK_OVERLAP,
    FIELD_GENERATE_SUMMARY,
    FIELD_EMBEDDING_DIMENSION,
    FIELD_TASK_ID,
    RESP_COLLECTIONS,
    RESP_TASK_ID,
    RESP_STATE,
    RESP_RESULT,
    RESP_FAILED_DOCUMENTS,
    RESP_VALIDATION_ERRORS,
    RESP_MESSAGE,
    RESP_ERROR,
)

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Indexes documents in vector store for RAG retrieval."""
    
    def __init__(self, base_url: str, timeout: int = 600):
        """Initialize document indexer."""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._created_collections: set = set()
        
        logger.info(f"DocumentIndexer initialized: {self.base_url}")
    
    def ensure_collection_exists(self, collection_name: str) -> bool:
        """Create collection if it doesn't exist."""
        if collection_name in self._created_collections:
            return True
        
        # Check if collection exists
        try:
            response = requests.get(
                f'{self.base_url}{API_INGESTOR_COLLECTIONS}',
                timeout=TIMEOUT_DEFAULT
            )
        except requests.RequestException as e:
            logger.error(f"Error checking collections: {e}")
            return False
        
        if response.status_code == 200:
            collections = response.json().get(RESP_COLLECTIONS, [])
            if collection_name in collections:
                logger.info(f"Collection '{collection_name}' already exists")
                self._created_collections.add(collection_name)
                return True
        
        # Create collection
        logger.info(f"Creating collection '{collection_name}'...")
        try:
            create_response = requests.post(
                f'{self.base_url}{API_INGESTOR_COLLECTION}',
                json={
                    FIELD_COLLECTION_NAME: collection_name,
                    FIELD_EMBEDDING_DIMENSION: COLLECTION_EMBEDDING_DIMENSION,
                    'metadata_schema': []
                },
                headers={'Content-Type': 'application/json'},
                timeout=TIMEOUT_DEFAULT
            )
        except requests.RequestException as e:
            logger.error(f"Error creating collection: {e}")
            return False
        
        if create_response.status_code in [200, 201]:
            logger.info(f"✓ Collection '{collection_name}' created")
            self._created_collections.add(collection_name)
            return True
        
        logger.error(f"Failed to create collection: {create_response.status_code}")
        return False
    
    def upload(
        self,
        file_data: bytes,
        filename: str,
        collection: str,
        chunk_size: int = COLLECTION_CHUNK_SIZE,
        chunk_overlap: int = COLLECTION_CHUNK_OVERLAP
    ) -> Optional[str]:
        """Upload document to ingestor server."""
        if not self.ensure_collection_exists(collection):
            logger.error("Failed to ensure collection exists")
            return None
        
        content_type = self._get_content_type(filename)
        files = {'documents': (filename, file_data, content_type)}
        
        data_config = {
            FIELD_COLLECTION_NAME: collection,
            FIELD_BLOCKING: False,
            FIELD_SPLIT_OPTIONS: {
                FIELD_CHUNK_SIZE: chunk_size,
                FIELD_CHUNK_OVERLAP: chunk_overlap
            },
            FIELD_GENERATE_SUMMARY: False
        }
        
        logger.info(f"Uploading to collection: {collection}")
        try:
            response = requests.post(
                f'{self.base_url}{API_INGESTOR_DOCUMENTS}',
                files=files,
                data={'data': json.dumps(data_config)},
                timeout=self.timeout
            )
        except requests.RequestException as e:
            logger.error(f"Error uploading document: {e}")
            return None
        
        if response.status_code in [200, 201, 202]:
            result = response.json()
            task_id = result.get(RESP_TASK_ID)
            if task_id:
                logger.info(f"✓ File uploaded, task_id: {task_id}")
                return task_id
            logger.error("No task_id in response")
            return None
        
        logger.error(f"Upload failed: {response.status_code} - {response.text}")
        return None
    
    def check_status(self, task_id: str, max_wait: int = TIMEOUT_MAX_TASK_WAIT) -> Tuple[bool, str]:
        """Check task status and wait for completion."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(
                    f'{self.base_url}{API_INGESTOR_STATUS}',
                    params={FIELD_TASK_ID: task_id},
                    timeout=TIMEOUT_DEFAULT
                )
            except requests.RequestException as e:
                return False, str(e)
            
            if response.status_code != 200:
                return False, f"Status check failed: {response.status_code}"
            
            result = response.json()
            state = result.get(RESP_STATE, 'UNKNOWN')
            
            if state == STATUS_FAILED:
                return False, result.get(RESP_ERROR, 'Unknown error')
            
            if state == STATUS_FINISHED:
                return self._parse_finished_result(result)
            
            if state in [STATUS_PENDING, STATUS_PROCESSING]:
                elapsed = int(time.time() - start_time)
                if elapsed % 5 == 0:
                    logger.info(f"Task {task_id}: {state} ({elapsed}s)")
            
            time.sleep(1)
        
        return False, f"Timeout after {max_wait}s"
    
    def _parse_finished_result(self, result: dict) -> Tuple[bool, str]:
        """Parse result from a finished task."""
        task_result = result.get(RESP_RESULT, {})
        failed_docs = task_result.get(RESP_FAILED_DOCUMENTS, [])
        validation_errors = task_result.get(RESP_VALIDATION_ERRORS, [])
        
        if failed_docs or validation_errors:
            return False, f"Failed: {failed_docs}, Errors: {validation_errors}"
        return True, task_result.get(RESP_MESSAGE, 'Completed')
    
    def delete(self, filename: str, collection: str) -> bool:
        """Delete document from collection."""
        logger.info(f"Deleting '{filename}' from '{collection}'")
        
        try:
            response = requests.delete(
                f'{self.base_url}{API_INGESTOR_DOCUMENTS}',
                params={FIELD_COLLECTION_NAME: collection},
                json=[filename],
                headers={'Content-Type': 'application/json'},
                timeout=TIMEOUT_DEFAULT
            )
        except requests.RequestException as e:
            logger.error(f"Error deleting document: {e}")
            return False
        
        if response.status_code in [200, 201, 204]:
            logger.info(f"Deleted '{filename}'")
            return True
        
        logger.error(f"Delete failed: {response.status_code}")
        return False
    
    def index_video_description(
        self,
        collection: str,
        video_id: str,
        video_name: str,
        description: str
    ) -> bool:
        """Index video description in Milvus."""
        try:
            # Ensure collection exists first
            if not self.ensure_collection_exists(collection):
                logger.error(f"Failed to ensure collection '{collection}' exists")
                return False
            
            doc_filename = f"{video_name}_description.json"
            
            # Delete existing document first to allow re-indexing
            self.delete(doc_filename, collection)
            
            content = f"""Video Name: {video_name}
Video ID: {video_id}
Source: VSS Analysis

Description:
{description}"""
            
            doc_data = {
                'content': content,
                'metadata': {
                    'content_type': 'video',
                    'video_id': video_id,
                    'video_name': video_name,
                    'source': 'vss_analysis'
                }
            }
            
            files = {
                'documents': (doc_filename, json.dumps(doc_data).encode(), 'application/json')
            }
            
            data_config = {
                FIELD_COLLECTION_NAME: collection,
                FIELD_BLOCKING: True,
                FIELD_SPLIT_OPTIONS: {
                    FIELD_CHUNK_SIZE: COLLECTION_VIDEO_CHUNK_SIZE,
                    FIELD_CHUNK_OVERLAP: COLLECTION_VIDEO_CHUNK_OVERLAP
                },
                FIELD_GENERATE_SUMMARY: False
            }
            
            response = requests.post(
                f'{self.base_url}{API_INGESTOR_DOCUMENTS}',
                files=files,
                data={'data': json.dumps(data_config)},
                timeout=VIDEO_INDEX_TIMEOUT
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"✓ Video description indexed to '{collection}'")
                return True
            else:
                logger.error(f"Failed to index video description: {response.status_code} - {response.text}")
                return False
            
        except requests.RequestException as e:
            logger.error(f"Error indexing video description: {e}")
            return False
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type from filename."""
        ext = Path(filename).suffix.lower()
        return CONTENT_TYPE_MAP.get(ext, DEFAULT_CONTENT_TYPE)

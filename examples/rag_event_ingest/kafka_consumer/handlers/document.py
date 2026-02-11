# handlers/document.py
"""Handler for document files (PDF, DOCX, TXT, etc.)."""

import logging

import requests

from .base import BaseHandler
from models.events import S3Event, HandlerResult
from services.storage import ObjectStorage
from services.document_indexer import DocumentIndexer

logger = logging.getLogger(__name__)


class DocumentHandler(BaseHandler):
    """Handler for document files - sends to RAG ingestor."""
    
    def __init__(self, storage: ObjectStorage, indexer: DocumentIndexer):
        """Initialize document handler.
        
        Args:
            storage: Object storage for file downloads
            indexer: Document indexer for RAG pipeline
        """
        self.storage = storage
        self.indexer = indexer
    
    @property
    def name(self) -> str:
        return "DocumentHandler"
    
    def handle(self, event: S3Event) -> HandlerResult:
        """Process document file.
        
        1. Delete existing entries (for updates)
        2. Download from MinIO
        3. Upload to ingestor
        4. Wait for completion
        
        Args:
            event: S3 event with document info
            
        Returns:
            HandlerResult with task_id for status tracking
        """
        self.log_start(event)
        
        try:
            # Step 1: Delete existing entries (handles updates)
            logger.info(f"🔄 Checking for existing entries of {event.key}...")
            self.indexer.delete(event.key, event.collection)
            
            # Step 2: Download from storage
            logger.info(f"📥 Downloading from storage...")
            file_data = self.storage.download(event.bucket, event.key)
            
            # Step 3: Upload to indexer
            logger.info(f"📤 Sending to indexer...")
            task_id = self.indexer.upload(
                file_data=file_data,
                filename=event.key,
                collection=event.collection
            )
            
            if not task_id:
                result = HandlerResult.failed_result("Indexer upload failed")
                self.log_failure(event, result)
                return result
            
            # Step 4: Wait for completion
            logger.info(f"⏳ Waiting for indexing (task_id: {task_id})...")
            success, message = self.indexer.check_status(task_id)
            
            if success:
                result = HandlerResult.success_result(task_id=task_id)
                self.log_success(event, result)
                return result
            else:
                result = HandlerResult.failed_result(message, task_id=task_id)
                self.log_failure(event, result)
                return result
                
        except requests.RequestException as e:
            logger.error(f"Network error processing document: {e}")
            return HandlerResult.failed_result(str(e))
        except (IOError, OSError) as e:
            logger.error(f"Storage error processing document: {e}")
            return HandlerResult.failed_result(str(e))

# handlers/video.py
"""Handler for video files."""

import logging

import requests

from .base import BaseHandler
from models.events import S3Event, HandlerResult
from services.storage import ObjectStorage
from services.video_analyzer import VideoAnalyzer
from services.document_indexer import DocumentIndexer

logger = logging.getLogger(__name__)


class VideoHandler(BaseHandler):
    """Handler for video files - analyzes and indexes in vector store."""
    
    def __init__(
        self,
        storage: ObjectStorage,
        analyzer: VideoAnalyzer,
        indexer: DocumentIndexer,
        enable_multimodal_rag: bool = True
    ):
        """Initialize video handler.
        
        Args:
            storage: Object storage for file downloads
            analyzer: Video analyzer for VLM analysis
            indexer: Document indexer for storing descriptions
            enable_multimodal_rag: Whether to index video descriptions
        """
        self.storage = storage
        self.analyzer = analyzer
        self.indexer = indexer
        self.enable_multimodal_rag = enable_multimodal_rag
    
    @property
    def name(self) -> str:
        return "VideoHandler"
    
    def handle(self, event: S3Event) -> HandlerResult:
        """Process video file.
        
        1. Download from MinIO
        2. Upload to VSS
        3. (Optional) Get description and index in Milvus
        
        Args:
            event: S3 event with video info
            
        Returns:
            HandlerResult
        """
        self.log_start(event)
        
        try:
            # Step 1: Download from storage
            logger.info(f"📥 Downloading video from storage...")
            video_data = self.storage.download(event.bucket, event.key)
            
            # Step 2: Upload to VSS
            logger.info(f"📤 Uploading to analyzer...")
            success, video_name = self.analyzer.upload_video(video_data, event.key)
            
            if not success:
                result = HandlerResult.failed_result("Video upload failed")
                self.log_failure(event, result)
                return result
            
            logger.info(f"✓ Video uploaded: {video_name}")
            
            # Step 3: Multi-modal RAG indexing (optional)
            if self.enable_multimodal_rag and video_name:
                self._index_video_description(event, video_name)
            
            result = HandlerResult.success_result()
            self.log_success(event, result)
            return result
            
        except requests.RequestException as e:
            logger.error(f"Network error processing video: {e}")
            return HandlerResult.failed_result(str(e))
        except (IOError, OSError) as e:
            logger.error(f"Storage error processing video: {e}")
            return HandlerResult.failed_result(str(e))
    
    def _index_video_description(self, event: S3Event, video_name: str):
        """Get video description from VSS and index in Milvus.
        
        Args:
            event: S3 event
            video_name: Video name (stem of filename)
        """
        logger.info(f"🔄 Starting Multi-Modal RAG indexing...")
        
        # Get description using VSS /generate with detailed prompt
        logger.info(f"📹 Generating description for video: {video_name}...")
        description = self.analyzer.get_video_description(video_name)
        
        if not description:
            logger.warning("Failed to get video description from VSS")
            return
        
        # Index description in Milvus
        success = self.indexer.index_video_description(
            collection=event.collection,
            video_id=video_name,
            video_name=event.key,
            description=description
        )
        
        if success:
            logger.info(f"✓ Indexed video description for {event.key}")
        else:
            logger.warning(f"⚠ Failed to index video description")

# handlers/video.py
"""Handler for video files using LVS API.

Flow: Download from S3 -> Upload to media storage -> LVS /summarize -> index.
"""

import logging
from pathlib import PurePosixPath

import requests

from .base import BaseHandler
from models.events import S3Event, HandlerResult
from services.storage import ObjectStorage
from services.video_analyzer import VideoAnalyzer
from services.document_indexer import DocumentIndexer

logger = logging.getLogger(__name__)


class VideoHandler(BaseHandler):
    """Handler for video files -- LVS flow.

    Downloads video from S3, uploads to media storage (so LVS can
    access it internally), then calls LVS /summarize and indexes
    the result.
    """

    def __init__(
        self,
        storage: ObjectStorage,
        analyzer: VideoAnalyzer,
        indexer: DocumentIndexer,
        enable_multimodal_rag: bool = True,
    ):
        self.storage = storage
        self.analyzer = analyzer
        self.indexer = indexer
        self.enable_multimodal_rag = enable_multimodal_rag

    @property
    def name(self) -> str:
        return "VideoHandler"

    def handle(self, event: S3Event) -> HandlerResult:
        """Download -> upload to storage -> LVS summarize -> index.

        Args:
            event: S3 event with video info

        Returns:
            HandlerResult
        """
        self.log_start(event)

        try:
            video_data = self.storage.download(event.bucket, event.key)
            logger.info(f"Downloaded {event.bucket}/{event.key} ({len(video_data)} bytes)")

            video_url = self.analyzer.upload_video(video_data, event.key)
            if not video_url:
                result = HandlerResult.failed_result("Failed to upload video to media storage")
                self.log_failure(event, result)
                return result

            description = self.analyzer.summarize(video_url)
            if not description:
                result = HandlerResult.failed_result("LVS summarization returned no content")
                self.log_failure(event, result)
                return result

            if self.enable_multimodal_rag:
                self._index_video_description(event, description)

            result = HandlerResult.success_result()
            self.log_success(event, result)
            return result

        except requests.RequestException as e:
            logger.error(f"Network error processing video: {e}")
            return HandlerResult.failed_result(str(e))
        except (IOError, OSError) as e:
            logger.error(f"Storage error processing video: {e}")
            return HandlerResult.failed_result(str(e))

    def _index_video_description(self, event: S3Event, description: str):
        """Index video description in vector store."""
        video_name = PurePosixPath(event.key).stem
        success = self.indexer.index_video_description(
            collection=event.collection,
            video_id=video_name,
            video_name=event.key,
            description=description,
        )
        if success:
            logger.info(f"Indexed video description for {event.key}")
        else:
            logger.warning(f"Failed to index video description for {event.key}")

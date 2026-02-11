# router.py
"""File routing module for MinIO event processing."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Union

from config.constants import (
    VIDEO_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    AUDIO_EXTENSIONS,
    SKIP_EXTENSIONS,
    DEST_VSS,
    DEST_RAG,
    DEST_SKIP,
    KEY_DESTINATION,
    KEY_FILE_TYPE,
    KEY_EXTENSION,
    KEY_REASON,
    FILE_TYPE_VIDEO,
    FILE_TYPE_DOCUMENT,
    FILE_TYPE_IMAGE,
    FILE_TYPE_AUDIO,
    FILE_TYPE_SKIP,
    FILE_TYPE_UNKNOWN,
    CFG_VIDEO_EXTENSIONS,
    CFG_DOCUMENT_EXTENSIONS,
    CFG_IMAGE_EXTENSIONS,
    CFG_AUDIO_EXTENSIONS,
    CFG_SKIP_EXTENSIONS,
    CFG_ENABLE_IMAGE_PROCESSING,
    CFG_ENABLE_AUDIO_PROCESSING,
)

logger = logging.getLogger(__name__)


class FileRouter:
    """Routes files to appropriate processing services based on file type."""
    
    def __init__(self, config: Union[Dict[str, Any], Any] = None):
        """Initialize router with optional config overrides."""
        if config is None:
            config = {}
        elif hasattr(config, '__dataclass_fields__'):
            config = {
                CFG_VIDEO_EXTENSIONS: config.video_extensions,
                CFG_DOCUMENT_EXTENSIONS: config.document_extensions,
                CFG_IMAGE_EXTENSIONS: config.image_extensions,
                CFG_AUDIO_EXTENSIONS: config.audio_extensions,
                CFG_SKIP_EXTENSIONS: config.skip_extensions,
                CFG_ENABLE_IMAGE_PROCESSING: config.enable_image_processing,
                CFG_ENABLE_AUDIO_PROCESSING: config.enable_audio_processing,
            }
        
        self.config = config
        self.video_extensions = self._to_set(config.get(CFG_VIDEO_EXTENSIONS, VIDEO_EXTENSIONS))
        self.document_extensions = self._to_set(config.get(CFG_DOCUMENT_EXTENSIONS, DOCUMENT_EXTENSIONS))
        self.image_extensions = self._to_set(config.get(CFG_IMAGE_EXTENSIONS, IMAGE_EXTENSIONS))
        self.audio_extensions = self._to_set(config.get(CFG_AUDIO_EXTENSIONS, AUDIO_EXTENSIONS))
        self.skip_extensions = self._to_set(config.get(CFG_SKIP_EXTENSIONS, SKIP_EXTENSIONS))
        self.enable_image_processing = config.get(CFG_ENABLE_IMAGE_PROCESSING, False)
        self.enable_audio_processing = config.get(CFG_ENABLE_AUDIO_PROCESSING, False)
        
        logger.info(f"FileRouter initialized - Video: {len(self.video_extensions)} types, "
                    f"Documents: {len(self.document_extensions)} types")
    
    @staticmethod
    def _to_set(value: Union[List, Set, None]) -> Set[str]:
        if value is None:
            return set()
        return set(value) if isinstance(value, (list, tuple)) else value
    
    def route(self, filename: str) -> dict:
        """Determine routing destination for a file."""
        ext = Path(filename).suffix.lower()
        
        if ext in self.skip_extensions:
            return {KEY_DESTINATION: DEST_SKIP, KEY_FILE_TYPE: FILE_TYPE_SKIP, KEY_EXTENSION: ext, KEY_REASON: 'File extension in skip list'}
        
        if ext in self.video_extensions:
            return {KEY_DESTINATION: DEST_VSS, KEY_FILE_TYPE: FILE_TYPE_VIDEO, KEY_EXTENSION: ext}
        
        if ext in self.document_extensions:
            return {KEY_DESTINATION: DEST_RAG, KEY_FILE_TYPE: FILE_TYPE_DOCUMENT, KEY_EXTENSION: ext}
        
        if ext in self.image_extensions:
            if self.enable_image_processing:
                return {KEY_DESTINATION: DEST_RAG, KEY_FILE_TYPE: FILE_TYPE_IMAGE, KEY_EXTENSION: ext}
            return {KEY_DESTINATION: DEST_SKIP, KEY_FILE_TYPE: FILE_TYPE_IMAGE, KEY_EXTENSION: ext, KEY_REASON: 'Image processing not enabled'}
        
        if ext in self.audio_extensions:
            if self.enable_audio_processing:
                return {KEY_DESTINATION: DEST_RAG, KEY_FILE_TYPE: FILE_TYPE_AUDIO, KEY_EXTENSION: ext}
            return {KEY_DESTINATION: DEST_SKIP, KEY_FILE_TYPE: FILE_TYPE_AUDIO, KEY_EXTENSION: ext, KEY_REASON: 'Audio processing not enabled'}
        
        return {KEY_DESTINATION: DEST_RAG, KEY_FILE_TYPE: FILE_TYPE_UNKNOWN, KEY_EXTENSION: ext, KEY_REASON: 'Unknown extension, attempting RAG ingestion'}
    
    def is_video(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.video_extensions
    
    def is_document(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.document_extensions

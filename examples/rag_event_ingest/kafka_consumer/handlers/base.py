# handlers/base.py
"""Base handler abstract class."""

from abc import ABC, abstractmethod
import logging

from models.events import S3Event, HandlerResult

logger = logging.getLogger(__name__)


class BaseHandler(ABC):
    """Abstract base class for file handlers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Handler name for logging."""
        pass
    
    @abstractmethod
    def handle(self, event: S3Event) -> HandlerResult:
        """Process an S3 event.
        
        Args:
            event: S3 event to process
            
        Returns:
            HandlerResult with success status and optional task_id
        """
        pass
    
    def log_start(self, event: S3Event):
        """Log handler start."""
        logger.info(f"[{self.name}] Processing {event.bucket}/{event.key}")
    
    def log_success(self, event: S3Event, result: HandlerResult):
        """Log successful handling."""
        logger.info(f"[{self.name}] ✓ {event.key} → {result.status}")
    
    def log_failure(self, event: S3Event, result: HandlerResult):
        """Log failed handling."""
        logger.error(f"[{self.name}] ✗ {event.key}: {result.error_message}")

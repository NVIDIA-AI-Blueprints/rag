# models/events.py
"""Data models for Kafka consumer events and results."""

from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Callable, ClassVar, Dict, Optional
from urllib.parse import unquote_plus

from config.constants import (
    STATUS_SUCCESS,
    STATUS_FAILED,
    STATUS_SKIPPED,
    # S3 Event fields
    EVENT_NAME,
    EVENT_RECORDS,
    EVENT_FIRST_RECORD_INDEX,
    EVENT_S3,
    EVENT_BUCKET,
    EVENT_OBJECT,
    EVENT_KEY,
    EVENT_SIZE,
    EVENT_ETAG,
    EVENT_NAME_FIELD,
    EVENT_PREFIX_CREATED,
    EVENT_PREFIX_REMOVED,
    EVENT_TYPE_CREATE,
    EVENT_TYPE_DELETE,
    # Record field names (for transformers)
    FIELD_START_TIME,
    FIELD_END_TIME,
    FIELD_DURATION_SECONDS,
)


@dataclass
class S3Event:
    """Represents a MinIO S3 event from Kafka."""
    bucket: str
    key: str
    size: int
    etag: str
    event_type: str
    collection: str = ''
    
    @classmethod
    def from_kafka_message(
        cls,
        event: Dict[str, Any],
        collection_resolver: Callable[[str], str]
    ) -> Optional['S3Event']:
        """Parse S3 event from Kafka message.
        
        Args:
            event: Raw Kafka message value
            collection_resolver: Function to resolve bucket -> collection name
        """
        if EVENT_NAME not in event:
            return None
        
        event_name = event[EVENT_NAME]
        
        if event_name.startswith(EVENT_PREFIX_CREATED):
            event_type = EVENT_TYPE_CREATE
        elif event_name.startswith(EVENT_PREFIX_REMOVED):
            event_type = EVENT_TYPE_DELETE
        else:
            return None
        
        records = event.get(EVENT_RECORDS, [])
        if not records:
            return None
        
        record = records[EVENT_FIRST_RECORD_INDEX]
        s3_data = record[EVENT_S3]
        bucket = s3_data[EVENT_BUCKET][EVENT_NAME_FIELD]
        obj_data = s3_data[EVENT_OBJECT]
        key = unquote_plus(obj_data[EVENT_KEY])
        size = obj_data.get(EVENT_SIZE, 0)
        etag = obj_data.get(EVENT_ETAG, '')
        
        return cls(
            bucket=bucket,
            key=key,
            size=size,
            etag=etag,
            event_type=event_type,
            collection=collection_resolver(bucket)
        )


@dataclass
class HandlerResult:
    """Result from a handler execution."""
    success: bool
    status: str  # SUCCESS, FAILED, SKIPPED, DELETED
    error_message: Optional[str] = None
    task_id: Optional[str] = None  # For RAG status tracking
    
    @classmethod
    def success_result(cls, task_id: Optional[str] = None) -> 'HandlerResult':
        return cls(success=True, status=STATUS_SUCCESS, task_id=task_id)
    
    @classmethod
    def failed_result(cls, error: str, task_id: Optional[str] = None) -> 'HandlerResult':
        return cls(success=False, status=STATUS_FAILED, error_message=error, task_id=task_id)
    
    @classmethod
    def skipped_result(cls, reason: str) -> 'HandlerResult':
        return cls(success=True, status=STATUS_SKIPPED, error_message=reason)


@dataclass
class IngestionRecord:
    """Record of an ingestion operation for history tracking."""
    file_name: str
    bucket: str
    collection: str
    status: str
    start_time: datetime
    end_time: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    task_id: Optional[str] = None
    
    _TRANSFORMERS: ClassVar[Dict[str, Callable]] = {
        FIELD_START_TIME: lambda v: v.isoformat(),
        FIELD_END_TIME: lambda v: v.isoformat(),
        FIELD_DURATION_SECONDS: lambda v: round(v, 2),
    }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            transform = self._TRANSFORMERS.get(f.name)
            result[f.name] = transform(value) if transform else value
        return result

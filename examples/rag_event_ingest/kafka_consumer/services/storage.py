# services/storage.py
"""S3-compatible object storage service."""

import io
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

from minio import Minio
from minio.error import S3Error

from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    MINIO_DEFAULT_COLLECTION,
    MINIO_SOURCES,
    CFG_ENDPOINT,
    CFG_ACCESS,
    CFG_SECRET,
    CFG_SECURE,
    CFG_COLLECTION,
    CFG_BUCKETS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Interface
# =============================================================================

class StorageBackend(ABC):
    """Abstract interface for object storage operations.
    
    Implement this to add new backends (Azure Blob, GCS, etc.)
    """
    
    @abstractmethod
    def download(self, bucket: str, key: str) -> bytes:
        """Download file from storage."""
        pass
    
    @abstractmethod
    def upload(self, bucket: str, key: str, data: bytes, content_type: Optional[str] = None) -> None:
        """Upload file to storage."""
        pass
    
    @abstractmethod
    def delete(self, bucket: str, key: str) -> None:
        """Delete file from storage."""
        pass
    
    @abstractmethod
    def exists(self, bucket: str, key: str) -> bool:
        """Check if file exists."""
        pass


# =============================================================================
# S3 Implementation
# =============================================================================

class S3Backend(StorageBackend):
    """S3-compatible storage (MinIO, AWS S3, Wasabi, etc.)."""
    
    def __init__(self, client: Minio):
        self._client = client
    
    @classmethod
    def create(
        cls,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
    ) -> 'S3Backend':
        """Factory method to create S3 backend."""
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        logger.info(f"Created S3 client: {endpoint}")
        return cls(client)
    
    def download(self, bucket: str, key: str) -> bytes:
        response = self._client.get_object(bucket, key)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()
        logger.info(f"Downloaded {bucket}/{key} ({len(data)} bytes)")
        return data
    
    def upload(self, bucket: str, key: str, data: bytes, content_type: Optional[str] = None) -> None:
        self._client.put_object(
            bucket, key, io.BytesIO(data),
            length=len(data),
            content_type=content_type or 'application/octet-stream'
        )
        logger.info(f"Uploaded {bucket}/{key}")
    
    def delete(self, bucket: str, key: str) -> None:
        self._client.remove_object(bucket, key)
        logger.info(f"Deleted {bucket}/{key}")
    
    def exists(self, bucket: str, key: str) -> bool:
        try:
            self._client.stat_object(bucket, key)
            return True
        except S3Error:
            return False


# =============================================================================
# Object Storage (Factory + Bucket Mapping)
# =============================================================================

class ObjectStorage:
    """Object storage with bucket-to-collection mapping.
    
    Handles single or multiple S3 sources via configuration.
    """
    
    def __init__(self):
        self._backends: Dict[str, StorageBackend] = {}
        self._bucket_to_backend: Dict[str, str] = {}
        self._bucket_to_collection: Dict[str, str] = {}
        self._default_collection = MINIO_DEFAULT_COLLECTION
        self._configure()
    
    def _configure(self):
        if MINIO_SOURCES:
            self._configure_multi_source(MINIO_SOURCES)
        else:
            self._configure_single_source()
    
    def _configure_single_source(self):
        logger.info(f"Single S3 mode: {MINIO_ENDPOINT}")
        self._backends['default'] = S3Backend.create(
            MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE
        )
    
    def _configure_multi_source(self, sources_json: str):
        config = json.loads(sources_json)
        for name, src in config.items():
            self._configure_source(name, src)
    
    def _configure_source(self, name: str, src: dict):
        """Configure a single S3 source and register its buckets."""
        logger.info(f"Configuring S3 source '{name}': {src[CFG_ENDPOINT]}")
        
        self._backends[name] = S3Backend.create(
            src[CFG_ENDPOINT],
            src.get(CFG_ACCESS, MINIO_ACCESS_KEY),
            src.get(CFG_SECRET, MINIO_SECRET_KEY),
            src.get(CFG_SECURE, False)
        )
        
        collection = src.get(CFG_COLLECTION, name.replace('-', '_'))
        self._register_buckets(name, src.get(CFG_BUCKETS, []), collection)
    
    def _register_buckets(self, backend_name: str, buckets: list, collection: str):
        """Register bucket-to-backend and bucket-to-collection mappings."""
        for bucket in buckets:
            self._bucket_to_backend[bucket] = backend_name
            self._bucket_to_collection[bucket] = collection
            logger.info(f"  {bucket} → {collection}")
    
    def _get_backend(self, bucket: str) -> StorageBackend:
        if bucket in self._bucket_to_backend:
            return self._backends[self._bucket_to_backend[bucket]]
        return next(iter(self._backends.values()))
    
    def download(self, bucket: str, key: str) -> bytes:
        return self._get_backend(bucket).download(bucket, key)
    
    def get_collection_for_bucket(self, bucket: str) -> str:
        """Get collection name for bucket.
        
        Priority:
        1. Explicit mapping from MINIO_SOURCES config
        2. Default collection from COLLECTION_NAME env var
        3. Fallback: bucket name with hyphens → underscores
        """
        # Check explicit mapping first
        if bucket in self._bucket_to_collection:
            return self._bucket_to_collection[bucket]
        
        # Use default collection if configured
        if self._default_collection:
            return self._default_collection
        
        # Fallback to bucket name conversion
        return bucket.replace('-', '_')

# config/constants.py
"""Static constants that don't change at runtime.

For configurable values from environment, see settings.py
"""

# ==================== File Extensions ====================

DOCUMENT_EXTENSIONS = frozenset({
    '.pdf', '.docx', '.doc', '.txt', '.md', '.rst',
    '.html', '.htm', '.pptx', '.ppt', '.xlsx', '.xls',
    '.csv', '.json', '.xml'
})

IMAGE_EXTENSIONS = frozenset({
    '.jpg', '.jpeg', '.png', '.gif', 
    '.webp', '.bmp', '.tiff', '.svg'
})

AUDIO_EXTENSIONS = frozenset({
    '.mp3', '.wav', '.flac', '.aac', 
    '.ogg', '.m4a', '.wma'
})

SKIP_EXTENSIONS = frozenset({
    '.tmp', '.log', '.bak', '.swp', '.DS_Store',
    '.gitkeep', '.gitignore'
})


# ==================== Content Types ====================

CONTENT_TYPE_MAP = {
    # Documents
    '.pdf': 'application/pdf',
    '.txt': 'text/plain',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.html': 'text/html',
    '.htm': 'text/html',
    '.xml': 'application/xml',
    '.json': 'application/json',
    '.csv': 'text/csv',
    '.md': 'text/markdown',
    '.rst': 'text/x-rst',
    '.ppt': 'application/vnd.ms-powerpoint',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.xls': 'application/vnd.ms-excel',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    # Images
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.bmp': 'image/bmp',
    '.tiff': 'image/tiff',
    '.svg': 'image/svg+xml',
    # Audio
    '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav',
    '.flac': 'audio/flac',
    '.aac': 'audio/aac',
    '.ogg': 'audio/ogg',
    '.m4a': 'audio/mp4',
    '.wma': 'audio/x-ms-wma',
}

DEFAULT_CONTENT_TYPE = 'application/octet-stream'


# ==================== Routing ====================

# Destinations
DEST_RAG = 'rag'
DEST_SKIP = 'skip'
DEST_UNKNOWN = 'unknown'

# Route result keys
KEY_DESTINATION = 'destination'
KEY_FILE_TYPE = 'file_type'
KEY_EXTENSION = 'extension'
KEY_REASON = 'reason'

# File types
FILE_TYPE_DOCUMENT = 'document'
FILE_TYPE_IMAGE = 'image'
FILE_TYPE_AUDIO = 'audio'
FILE_TYPE_SKIP = 'skip'
FILE_TYPE_UNKNOWN = 'unknown'

# Config keys
CFG_DOCUMENT_EXTENSIONS = 'document_extensions'
CFG_IMAGE_EXTENSIONS = 'image_extensions'
CFG_AUDIO_EXTENSIONS = 'audio_extensions'
CFG_SKIP_EXTENSIONS = 'skip_extensions'
CFG_ENABLE_IMAGE_PROCESSING = 'enable_image_processing'
CFG_ENABLE_AUDIO_PROCESSING = 'enable_audio_processing'


# ==================== S3 Event Fields ====================

# Kafka S3 event structure
EVENT_NAME = 'EventName'
EVENT_RECORDS = 'Records'
EVENT_FIRST_RECORD_INDEX = 0  # S3 events typically contain single record
EVENT_S3 = 's3'
EVENT_BUCKET = 'bucket'
EVENT_OBJECT = 'object'
EVENT_KEY = 'key'
EVENT_SIZE = 'size'
EVENT_ETAG = 'eTag'
EVENT_NAME_FIELD = 'name'

# Event type prefixes
EVENT_PREFIX_CREATED = 's3:ObjectCreated:'
EVENT_PREFIX_REMOVED = 's3:ObjectRemoved:'

# Event type values
EVENT_TYPE_CREATE = 'create'
EVENT_TYPE_DELETE = 'delete'


# ==================== Record Fields ====================

# IngestionRecord field names (dataclass attributes)
FIELD_FILE_NAME = 'file_name'
FIELD_BUCKET = 'bucket'
FIELD_COLLECTION = 'collection'
FIELD_STATUS = 'status'
FIELD_START_TIME = 'start_time'
FIELD_END_TIME = 'end_time'
FIELD_DURATION_SECONDS = 'duration_seconds'
FIELD_ERROR_MESSAGE = 'error_message'
FIELD_TASK_ID = 'task_id'

# IngestionRecord serialization output keys
RECORD_FILE_NAME = FIELD_FILE_NAME
RECORD_BUCKET = FIELD_BUCKET
RECORD_COLLECTION = FIELD_COLLECTION
RECORD_START_TIME = FIELD_START_TIME
RECORD_END_TIME = FIELD_END_TIME
RECORD_DURATION = FIELD_DURATION_SECONDS
RECORD_STATUS = FIELD_STATUS
RECORD_ERROR = FIELD_ERROR_MESSAGE
RECORD_TASK_ID = FIELD_TASK_ID


# ==================== Task Status ====================

STATUS_PENDING = 'PENDING'
STATUS_PROCESSING = 'PROCESSING'
STATUS_FINISHED = 'FINISHED'
STATUS_FAILED = 'FAILED'
STATUS_SKIPPED = 'SKIPPED'
STATUS_DELETED = 'DELETED'
STATUS_SUCCESS = 'SUCCESS'


# ==================== Config Keys ====================

# MinIO/S3 source config keys
CFG_ENDPOINT = 'endpoint'
CFG_ACCESS = 'access'
CFG_SECRET = 'secret'
CFG_SECURE = 'secure'
CFG_COLLECTION = 'collection'
CFG_BUCKETS = 'buckets'


# ==================== API Request Fields ====================

# Ingestor request fields
FIELD_COLLECTION_NAME = 'collection_name'
FIELD_BLOCKING = 'blocking'
FIELD_SPLIT_OPTIONS = 'split_options'
FIELD_CHUNK_SIZE = 'chunk_size'
FIELD_CHUNK_OVERLAP = 'chunk_overlap'
FIELD_GENERATE_SUMMARY = 'generate_summary'
FIELD_EMBEDDING_DIMENSION = 'embedding_dimension'
FIELD_TASK_ID = 'task_id'


# ==================== API Response Fields ====================

# Common response fields
RESP_CONTENT = 'content'
RESP_RESPONSE = 'response'
RESP_TEXT = 'text'
RESP_CHOICES = 'choices'
RESP_MESSAGE = 'message'
RESP_ERROR = 'error'

# Ingestor response fields
RESP_COLLECTIONS = 'collections'
RESP_TASK_ID = 'task_id'
RESP_STATE = 'state'
RESP_RESULT = 'result'
RESP_FAILED_DOCUMENTS = 'failed_documents'
RESP_VALIDATION_ERRORS = 'validation_errors'


# ==================== Timeouts (seconds) ====================

TIMEOUT_DEFAULT = 30
TIMEOUT_UPLOAD = 600
TIMEOUT_TASK_CHECK = 30
TIMEOUT_MAX_TASK_WAIT = 300


# ==================== Kafka Defaults ====================

KAFKA_DEFAULT_TOPIC = 'aidp-topic'
KAFKA_DEFAULT_CONSUMER_GROUP = 'nvingest-consumer-group'
KAFKA_DEFAULT_AUTO_OFFSET_RESET = 'earliest'
KAFKA_DEFAULT_MAX_POLL_RECORDS = 1
KAFKA_DEFAULT_MAX_POLL_INTERVAL_MS = 600000   # 10 min
KAFKA_DEFAULT_SESSION_TIMEOUT_MS = 60000      # 60s
KAFKA_DEFAULT_HEARTBEAT_INTERVAL_MS = 20000   # 20s


# ==================== Collection Defaults ====================

COLLECTION_EMBEDDING_DIMENSION = 2048
COLLECTION_CHUNK_SIZE = 512
COLLECTION_CHUNK_OVERLAP = 150


# ==================== Environment Variable Keys ====================

# Kafka
ENV_KAFKA_BOOTSTRAP_SERVERS = 'KAFKA_BOOTSTRAP_SERVERS'
ENV_KAFKA_TOPIC = 'KAFKA_TOPIC'
ENV_CONSUMER_GROUP = 'CONSUMER_GROUP'
ENV_KAFKA_AUTO_OFFSET_RESET = 'KAFKA_AUTO_OFFSET_RESET'
ENV_KAFKA_MAX_POLL_RECORDS = 'KAFKA_MAX_POLL_RECORDS'
ENV_KAFKA_MAX_POLL_INTERVAL_MS = 'KAFKA_MAX_POLL_INTERVAL_MS'
ENV_KAFKA_SESSION_TIMEOUT_MS = 'KAFKA_SESSION_TIMEOUT_MS'
ENV_KAFKA_HEARTBEAT_INTERVAL_MS = 'KAFKA_HEARTBEAT_INTERVAL_MS'

# Service URLs
ENV_INGESTOR_SERVER_URL = 'INGESTOR_SERVER_URL'
ENV_INGESTOR_TIMEOUT = 'INGESTOR_TIMEOUT'

# API Endpoints
ENV_API_INGESTOR_DOCUMENTS = 'API_INGESTOR_DOCUMENTS'
ENV_API_INGESTOR_COLLECTIONS = 'API_INGESTOR_COLLECTIONS'
ENV_API_INGESTOR_COLLECTION = 'API_INGESTOR_COLLECTION'
ENV_API_INGESTOR_STATUS = 'API_INGESTOR_STATUS'

# MinIO
ENV_MINIO_ENDPOINT = 'MINIO_ENDPOINT'
ENV_MINIO_ACCESS_KEY = 'MINIO_ACCESS_KEY'
ENV_MINIO_SECRET_KEY = 'MINIO_SECRET_KEY'
ENV_MINIO_SECURE = 'MINIO_SECURE'
ENV_COLLECTION_NAME = 'COLLECTION_NAME'
ENV_MINIO_SOURCES = 'MINIO_SOURCES'

# Feature Flags
ENV_ENABLE_IMAGE_PROCESSING = 'ENABLE_IMAGE_PROCESSING'
ENV_ENABLE_AUDIO_PROCESSING = 'ENABLE_AUDIO_PROCESSING'

# Collection Settings
ENV_EMBEDDING_DIMENSION = 'EMBEDDING_DIMENSION'
ENV_CHUNK_SIZE = 'CHUNK_SIZE'
ENV_CHUNK_OVERLAP = 'CHUNK_OVERLAP'

# Logging
ENV_LOG_LEVEL = 'LOG_LEVEL'
ENV_LOG_FORMAT = 'LOG_FORMAT'

# History
ENV_HISTORY_FILE = 'HISTORY_FILE'

# ==================== API Endpoint Defaults ====================

DEFAULT_API_INGESTOR_DOCUMENTS = '/v1/documents'
DEFAULT_API_INGESTOR_COLLECTIONS = '/v1/collections'
DEFAULT_API_INGESTOR_COLLECTION = '/v1/collection'
DEFAULT_API_INGESTOR_STATUS = '/v1/status'


# ==================== Logging Defaults ====================

DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# ==================== History Defaults ====================

DEFAULT_HISTORY_FILE = '/tmp/ingestion_history.jsonl'


# ==================== MinIO Defaults ====================

DEFAULT_COLLECTION_NAME = 'multimodal_data'

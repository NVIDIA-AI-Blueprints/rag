# Kafka MinIO Consumer

A modular, event-driven file ingestion system that processes files uploaded to MinIO and routes them to appropriate AI services (RAG for documents, VSS for videos).

## Architecture Overview

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   MinIO     │────▶│    Kafka    │────▶│   Consumer   │
│  (Storage)  │     │   (Events)  │     │   (Router)   │
└─────────────┘     └─────────────┘     └──────┬───────┘
                                               │
                         ┌─────────────────────┼─────────────────────┐
                         │                     │                     │
                         ▼                     ▼                     ▼
                  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
                  │  Document   │       │    Video    │       │    Skip     │
                  │   Handler   │       │   Handler   │       │  (Ignored)  │
                  └──────┬──────┘       └──────┬──────┘       └─────────────┘
                         │                     │
                         ▼                     ▼
                  ┌─────────────┐       ┌─────────────┐
                  │  Document   │       │    Video    │
                  │   Indexer   │       │   Analyzer  │
                  │   (RAG)     │       │    (VSS)    │
                  └─────────────┘       └─────────────┘
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                            ┌─────────────┐
                            │   Milvus    │
                            │  (Vectors)  │
                            └─────────────┘
```

## Project Structure

```
kafka-consumer/
├── main.py                 # Application entry point
├── consumer.py             # Kafka consumer loop & event processing
├── router.py               # File type routing logic
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build
│
├── config/                 # Global configuration
│   ├── constants.py        # Static values (extensions, keys, status)
│   └── settings.py         # Environment variables
│
├── handlers/               # File type handlers
│   ├── base.py             # Abstract base handler
│   ├── document.py         # Document ingestion handler
│   └── video.py            # Video ingestion handler
│
├── services/               # External service integrations
│   ├── config/             # Service-specific config
│   │   ├── constants.py    # API endpoints, timeouts
│   │   └── settings.py     # Service URLs from env
│   ├── object_storage.py   # MinIO client
│   ├── document_indexer.py # RAG ingestor API
│   └── video_analyzer.py   # VSS API
│
├── models/                 # Data models
│   └── events.py           # S3Event, HandlerResult, IngestionRecord
│
├── deploy/                 # Kubernetes manifests
│   └── deployment-multimodal.yaml
│
├── scripts/                # Utility scripts
│   ├── build_and_deploy.sh # Build & deploy to K8s
│   └── view_history.py     # View ingestion history
│
└── docs/                   # Documentation
    └── ARCHITECTURE.md     # Detailed architecture
```

## Components

### Entry Point (`main.py`)

Initializes all components with dependency injection:

```python
# Services
storage = ObjectStorage()
indexer = DocumentIndexer(cfg.INGESTOR_SERVER_URL)
analyzer = VideoAnalyzer(cfg.VSS_SERVER_URL)

# Handlers
handlers = {
    DEST_RAG: DocumentHandler(storage, indexer),
    DEST_VSS: VideoHandler(storage, analyzer, indexer),
}

# Consumer
consumer = KafkaEventConsumer(handlers=handlers, storage=storage)
consumer.run()
```

### Consumer (`consumer.py`)

- Connects to Kafka and polls for MinIO S3 events
- Parses events into `S3Event` objects
- Routes files using `FileRouter`
- Dispatches to appropriate handler
- Records ingestion history

### Router (`router.py`)

Routes files based on extension:

| File Type | Extensions | Destination |
|-----------|------------|-------------|
| Video | `.mp4`, `.mkv`, `.avi`, ... | VSS |
| Document | `.pdf`, `.docx`, `.txt`, ... | RAG |
| Image | `.jpg`, `.png`, ... | Skip (configurable) |
| Audio | `.mp3`, `.wav`, ... | Skip (configurable) |

### Handlers

#### `DocumentHandler`
1. Downloads file from MinIO
2. Uploads to RAG ingestor
3. Waits for task completion

#### `VideoHandler`
1. Downloads video from MinIO
2. Uploads to VSS for analysis
3. Gets VLM-generated description
4. Indexes description in Milvus (multi-modal RAG)

### Services

| Service | Purpose |
|---------|---------|
| `ObjectStorage` | MinIO client for file downloads |
| `DocumentIndexer` | RAG ingestor API (upload, status, delete) |
| `VideoAnalyzer` | VSS API (upload, generate description) |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BOOTSTRAP_SERVERS` | `my-cluster-kafka-bootstrap.rag:9092` | Kafka brokers |
| `KAFKA_TOPIC` | `aidp-topic` | Topic to consume |
| `CONSUMER_GROUP` | `nvingest-consumer-group` | Consumer group ID |
| `INGESTOR_SERVER_URL` | `http://ingestor-server:8082` | RAG ingestor URL |
| `VSS_SERVER_URL` | `http://vss-agent:8000` | VSS server URL |
| `MINIO_ENDPOINT` | `rag-minio:9000` | MinIO endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `ENABLE_MULTIMODAL_RAG` | `true` | Index video descriptions |
| `LOG_LEVEL` | `INFO` | Logging level |

### Multi-MinIO Configuration

For multiple MinIO instances, set `MINIO_SOURCES` as JSON:

```json
{
  "minio-1": {
    "endpoint": "minio.ns1.svc:9000",
    "access": "minioadmin",
    "secret": "minioadmin",
    "buckets": ["bucket-1"],
    "collection": "collection_1"
  }
}
```

## Development

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python main.py
```

### Docker Build

```bash
docker build -t kafka-consumer:latest .
```

### Deploy to Kubernetes

```bash
./scripts/build_and_deploy.sh
```

### View Logs

```bash
kubectl logs -f -n rag -l app=minio-nvingest-consumer
```

### View Ingestion History

```bash
python scripts/view_history.py
```

## Data Flow

1. **File Upload**: User uploads file to MinIO bucket
2. **Event**: MinIO sends S3 event to Kafka
3. **Consume**: Consumer receives event from Kafka
4. **Route**: Router determines file type and destination
5. **Handle**: Handler processes file:
   - **Document**: Upload to RAG → Index in Milvus
   - **Video**: Upload to VSS → Get description → Index in Milvus
6. **Record**: Save ingestion result to history

## Design Principles

- **Dependency Injection**: Services injected into handlers
- **Single Responsibility**: Each module has one purpose
- **Configuration as Code**: All magic strings in config modules
- **Clean Code**: No hardcoded values in business logic
- **Extensibility**: Add new handlers for new file types

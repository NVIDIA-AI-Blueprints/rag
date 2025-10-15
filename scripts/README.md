### Bulk Ingestion CLI — Quick Guide

- Purpose: Ease bulk upload of large datasets to the ingestor server in sequential batches with professional logging and task polling.
- Scope: Can optionally create and/or delete collections via flags; otherwise ensure the target collection already exists.

Prerequisites:
- Ingestor server running (default `http://localhost:8082`).
- Target collection created (see ingestion notebook/API if needed).

Install:
```bash
pip install -r scripts/requirements.txt
```

Run:
```bash
python scripts/batch_ingestion.py \
  --folder data/multimodal/ \
  --collection-name my_collection \
  --ingestor-host localhost \
  --ingestor-port 8082 \
  --upload-batch-size 100 \
  -v
```

#### Create/Delete collection (optional)

- Create the collection before uploading:
```bash
python scripts/batch_ingestion.py \
  --folder data/multimodal/ \
  --collection-name my_collection \
  --create_collection
```

- Delete the collection after the run completes:
```bash
python scripts/batch_ingestion.py \
  --folder data/multimodal/ \
  --collection-name my_collection \
  --delete_collection
```

- Create at start and delete at end (useful for temporary runs):
```bash
python scripts/batch_ingestion.py \
  --folder data/multimodal/ \
  --collection-name my_collection \
  --create_collection --delete_collection
```

Notes:
- Idempotent: Lists existing documents and skips already ingested files.
- Progress: Uploads sequential batches and shows progress (e.g., “Uploading batch 2/43 …”).
- Polling: Polls each ingestion task until it finishes before proceeding; fails the batch on FAILED/UNKNOWN/timeout.



### Retriever API CLI — Quick Guide

> Prerequisite: Before running any scripts in this folder, deploy and start the services by following the [Quickstart guide](../docs/deploy-docker-self-hosted.md). Once services are up (RAG server and Ingestor server), proceed with the commands below.

Use `scripts/retriever_api_usage.py` to talk to the RAG server.

- Install dependencies: `pip install -r scripts/requirements.txt`
- Default server: `http://localhost:8081`

#### Generate (default)
Generates an answer from your query and streams it to the terminal.
```bash
python scripts/retriever_api_usage.py "What is RAG?"
```

#### Search
Retrieves the top documents related to your query and returns them as JSON.
```bash
python scripts/retriever_api_usage.py --mode search "Tell me about Robert Frost's poems"
```

#### Specify collection name
Pass a collection name with your query (overrides the default `multimodal_data`).
```bash
# Generate
python scripts/retriever_api_usage.py \
  --payload-json '{"collection_names":["my_collection"]}' \
  "What is lion doing?"

# Search
python scripts/retriever_api_usage.py \
  --mode search \
  --payload-json '{"collection_names":["my_collection"]}' \
  "Tell me about Robert Frost's poems"
```

#### Use a payload
Customize parameters or provide messages via JSON.
```bash
# JSON string
python scripts/retriever_api_usage.py \
  --payload-json '{"messages":[{"role":"user","content":"Your query here"}]}'

# From file
python scripts/retriever_api_usage.py --mode search --payload-file scripts/payloads/search.json
```

#### Save output / change host
Save the response to a file or point to a different server URL.
```bash
python scripts/retriever_api_usage.py --mode search --output-json out/result.json "my query"
python scripts/retriever_api_usage.py --host http://my-host:8081 "my query"
```

Notes:
- Provide a query or include it in the payload. For `generate` without a query, your payload must include `messages`.
- Only fields you provide in the payload override defaults.

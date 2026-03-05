# Ingestion History Feature

## Overview

The Kafka consumer now tracks all file ingestion attempts with:
- **File name**: Name of the file processed
- **Start time**: When event was detected from Kafka
- **End time**: When ingestion completed (or failed)
- **Duration**: Total processing time in seconds
- **Status**: SUCCESS, FAILED, or DELETED
- **Error message**: Details if ingestion failed

## Storage

History is stored as JSON Lines format in: `/tmp/ingestion_history.jsonl`

Each line is a JSON record:
```json
{
  "file_name": "document.pdf",
  "bucket": "aidp-bucket",
  "start_time": "2025-11-07T09:15:30.123456",
  "end_time": "2025-11-07T09:15:45.789012",
  "duration_seconds": 15.67,
  "status": "SUCCESS",
  "error_message": null
}
```

## View History

### From Host Machine

```bash
# View last 50 records
python3 /localhome/local-anngu/aidp-helm/scripts/kafka-consumer/view_history.py

# View last 100 records
python3 /localhome/local-anngu/aidp-helm/scripts/kafka-consumer/view_history.py /tmp/ingestion_history.jsonl 100
```

### From Kubernetes Pod

```bash
# Copy history file from pod
kubectl cp rag/kafka-consumer-xxxxx:/tmp/ingestion_history.jsonl /tmp/ingestion_history.jsonl

# View it
python3 /localhome/local-anngu/aidp-helm/scripts/kafka-consumer/view_history.py /tmp/ingestion_history.jsonl
```

### Live Viewing

```bash
# Exec into pod
kubectl exec -it -n rag kafka-consumer-xxxxx -- bash

# View history inside pod
python3 << 'EOF'
import json
with open('/tmp/ingestion_history.jsonl') as f:
    for line in f:
        print(json.loads(line))
EOF
```

## Example Output

```
📊 INGESTION HISTORY (last 10 records)

========================================================================================================================
File Name                                Start Time           End Time             Duration     Status    
========================================================================================================================
document.pdf                             2025-11-07 09:00:15  2025-11-07 09:00:28  13.45s       ✅ SUCCESS
images.jpeg                              2025-11-07 09:00:35  2025-11-07 09:00:48  13.21s       ❌ FAILED
  └─ Error: No records with Embeddings to insert detected
test.txt                                 2025-11-07 09:01:02  2025-11-07 09:01:05  3.12s        ✅ SUCCESS
old-file.pdf                             2025-11-07 09:02:15  2025-11-07 09:02:16  1.05s        🗑️  DELETED
report.docx                              2025-11-07 09:03:20  2025-11-07 09:03:45  25.33s       ✅ SUCCESS
========================================================================================================================

📈 Summary:
   Total: 5 | ✅ Success: 3 | ❌ Failed: 1 | 🗑️  Deleted: 1
   Average Duration: 11.23s
```

## Configuration

Set custom history file location via environment variable:

```yaml
# In deployment
env:
- name: HISTORY_FILE
  value: "/data/ingestion_history.jsonl"
```

## Persistence

**Default:** History stored in `/tmp` (lost on pod restart)

**Persistent storage:** Mount a PVC to save history permanently:

```yaml
# Add to kafka-consumer-deployment.yaml
spec:
  template:
    spec:
      volumes:
      - name: history
        persistentVolumeClaim:
          claimName: consumer-history-pvc
      
      containers:
      - name: consumer
        volumeMounts:
        - name: history
          mountPath: /data
        env:
        - name: HISTORY_FILE
          value: "/data/ingestion_history.jsonl"
```

## Export History

```bash
# Copy from pod
kubectl cp rag/kafka-consumer-xxxxx:/tmp/ingestion_history.jsonl ./history.jsonl

# Convert to CSV
python3 << 'EOF'
import json
import csv

records = []
with open('history.jsonl') as f:
    for line in f:
        records.append(json.loads(line))

with open('history.csv', 'w', newline='') as csvfile:
    if records:
        writer = csv.DictWriter(csvfile, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

print("✓ Exported to history.csv")
EOF
```

## Monitoring

Check logs for ingestion summaries:

```bash
kubectl logs -n rag kafka-consumer-xxxxx | grep "INGESTION SUMMARY"
```

Example log output:
```
2025-11-07 09:00:28 - __main__ - INFO - ✓ INGESTION SUMMARY: document.pdf | Duration: 13.45s | Status: SUCCESS
2025-11-07 09:00:48 - __main__ - INFO - ✗ INGESTION SUMMARY: images.jpeg | Duration: 13.21s | Status: FAILED
```

sudo docker build -t nvcr.io/nvstaging/blueprint/aidp-kafka-consumer:v2.0.0 .
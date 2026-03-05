#!/bin/bash
set -e

# Get number of instances from parameter, default to 2
NUM_MINIO_INSTANCES=${1:-2}

# Validate input
if ! [[ "$NUM_MINIO_INSTANCES" =~ ^[0-9]+$ ]] || [ "$NUM_MINIO_INSTANCES" -lt 1 ]; then
    echo "Error: Please provide a valid number (1 or greater)"
    echo "Usage: $0 [number_of_instances]"
    echo "Example: $0 3    # Deploy consumer for 3 MinIO instances"
    echo "         $0      # Deploy consumer for 2 MinIO instances (default)"
    exit 1
fi

echo "========================================="
echo "Deploy Consumer for $NUM_MINIO_INSTANCES MinIO Instance(s)"
echo "========================================="
echo ""

# Delete existing deployment if present
kubectl delete deployment minio-multi-consumer -n rag 2>/dev/null || true

# Create ConfigMap with Python script
echo "Creating consumer script ConfigMap..."
kubectl create configmap consumer-script -n rag \
  --from-file=kafka_minio_consumer.py \
  --dry-run=client -o yaml | kubectl apply -f -

echo ""
echo "Generating MinIO sources configuration for $NUM_MINIO_INSTANCES instance(s)..."

# Generate MINIO_SOURCES JSON dynamically
MINIO_SOURCES_JSON="{"
for i in $(seq 1 $NUM_MINIO_INSTANCES); do
    if [ $i -gt 1 ]; then
        MINIO_SOURCES_JSON+=","
    fi
    MINIO_SOURCES_JSON+="
              \"minio-standalone-$i\": {
                \"endpoint\": \"minio.minio-standalone-$i.svc.cluster.local:9000\",
                \"access\": \"minioadmin\",
                \"secret\": \"minioadmin\",
                \"buckets\": [\"aidp-bucket-$i\"],
                \"collection\": \"aidp_bucket_$i\"
              }"
done
MINIO_SOURCES_JSON+="
            }"

echo ""
echo "Deploying consumer..."

# Deploy consumer that handles multiple MinIO instances
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio-nvingest-consumer
  namespace: rag
  labels:
    app: minio-nvingest-consumer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio-nvingest-consumer
  template:
    metadata:
      labels:
        app: minio-nvingest-consumer
    spec:
      initContainers:
      - name: install-deps
        image: python:3.11-slim
        command: ['sh', '-c']
        args:
        - |
          pip install --target=/deps kafka-python==2.0.2 minio==7.2.0 requests==2.31.0 requests-toolbelt==1.0.0
        volumeMounts:
        - name: deps
          mountPath: /deps
      containers:
      - name: consumer
        image: python:3.11-slim
        command: ['python', '-u', '/app/kafka_minio_consumer.py']
        env:
        - name: PYTHONPATH
          value: "/deps"
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "my-cluster-kafka-bootstrap.rag.svc.cluster.local:9092"
        - name: INGESTOR_SERVER_URL
          value: "http://ingestor-server.rag.svc.cluster.local:8082"
        - name: CONSUMER_GROUP
          value: "nvingest-consumer-group"
        - name: HISTORY_FILE
          value: "/tmp/ingestion_history.jsonl"
        # Multi-MinIO configuration in JSON format
        - name: MINIO_SOURCES
          value: |
            $MINIO_SOURCES_JSON
        volumeMounts:
        - name: script
          mountPath: /app
        - name: deps
          mountPath: /deps
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
      volumes:
      - name: script
        configMap:
          name: consumer-script
      - name: deps
        emptyDir: {}
      restartPolicy: Always
EOF

echo ""
echo "Waiting for pod to be ready..."
sleep 10
kubectl wait --for=condition=ready pod -l app=minio-nvingest-consumer -n rag --timeout=180s

echo ""
echo "========================================="
echo "✓ Consumer deployed for $NUM_MINIO_INSTANCES instance(s)!"
echo "========================================="
echo ""
echo "This single consumer handles events from:"
for i in $(seq 1 $NUM_MINIO_INSTANCES); do
    echo "  - minio-standalone-$i/aidp-bucket-$i → Collection: aidp_bucket_$i"
done
echo ""
echo "View logs:"
echo "  kubectl logs -f -n rag -l app=minio-multi-consumer"
echo ""
echo "Test uploads:"
for i in $(seq 1 $NUM_MINIO_INSTANCES); do
    echo "  kubectl exec -n rag rag-minio-mc -- mc cp /etc/hosts minio-standalone-$i/aidp-bucket-$i/test$i.txt"
done
echo ""


<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Troubleshoot NVIDIA RAG Blueprint

The following issues might arise when you work with the [NVIDIA RAG Blueprint](readme.md).


:::{note}
For the full list of known issues, see [Known Issues](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/CHANGELOG.md#all-known-issues)
:::

:::{tip}
To navigate this page more easily, click the outline button at the top of the page. ![outline-button](assets/outline-button.png)
:::



## Expected Deployment Times

Understanding typical deployment times can help you determine if your deployment is progressing normally or if there's an issue.

### Docker Deployments (Self-Hosted Models)

The expected timeline for a first-time deployment is the following:

- **Total time** – 15-30 minutes (can extend to 45+ minutes on slower networks)
- **Model download** – 10-20 minutes (largest component, no progress bar visible)
- **Service initialization** – 5-10 minutes

Factors that can affect the expected timeline include the following:**

  - Internet bandwidth (model downloads are ~100-150GB)
  - GPU type and availability
  - System resources (CPU, RAM, disk I/O)

The expected timeline for subsequent deployments is the following:

- 2-5 minutes total time
- Models are already cached
- Services start much sooner

### Docker Deployments (NVIDIA-Hosted Models)

The expected timeline for a first-time deployment is the following:

- 5-10 minutes
- Much faster because no model downloads are required

### Kubernetes/Helm Deployments

**First-time deployment:**
- **Total time:** 60-70 minutes
- **NIM cache downloads:** 40-50 minutes
- **Service initialization:** 10-15 minutes
- **Pod startup:** 5-10 minutes

**Subsequent deployments:**
The expected timeline for a first-time deployment is the following:

- 60-70 minutes total time 
- 40-50 minutes for NIM cache downloads 
- 10-15 minutes for service initialization 
- 5-10 minutes for Pod startup 

The expected timeline for a subsequent deployments is the following:

- 10-15 minutes
- Much faster because no model downloads are required

:::{tip}
If your deployment exceeds these time ranges significantly, check the monitoring commands in the relevant deployment guide or refer to "Monitoring Model Download Progress" in the following section.
:::



## Monitoring Model Download Progress

During first-time deployments, large models are downloaded without visible progress indicators. Here's how to monitor the process:

### Docker Deployments

**Check container logs:**
```bash
# Monitor NIM LLM service
docker logs -f nim-llm-ms

# Monitor embedding service
docker logs -f nemoretriever-embedding-ms

# Monitor ranking service
docker logs -f nemoretriever-ranking-ms
```

**Check disk usage to verify download progress:**
```bash
# Check cache directory size (grows as models download)
du -sh ~/.cache/model-cache/

# Watch disk usage in real-time
watch -n 10 'du -sh ~/.cache/model-cache/'
```

**Check container stats:**
```bash
# View resource usage and verify containers are active
docker stats nim-llm-ms nemoretriever-embedding-ms nemoretriever-ranking-ms
```

### Kubernetes/Helm Deployments

**Check NIMCache status:**
```bash
# View cache download status
kubectl get nimcache -n rag

# Watch cache status in real-time
kubectl get nimcache -n rag -w
```

**Check pod logs:**
```bash
# List pods
kubectl get pods -n rag

# View logs of a specific NIM pod
kubectl logs -f <nim-pod-name> -n rag

# View init container logs (where downloads occur)
kubectl logs <pod-name> -n rag -c <init-container-name>
```

**Check PVC usage:**
```bash
# View persistent volume claims
kubectl get pvc -n rag

# Describe a specific PVC to see capacity and usage
kubectl describe pvc <pvc-name> -n rag
```

**Check events for download progress:**
```bash
# View recent events sorted by time
kubectl get events -n rag --sort-by='.lastTimestamp'

# Watch events in real-time
kubectl get events -n rag -w
```

:::{note}
During model downloads, pods may appear stuck in "ContainerCreating" or "Init" state. This is normal behavior while large model files are being downloaded in the background. Use the monitoring commands above to verify that progress is being made.
:::



## NIM Container Permission Error During Model Download

When deploying self-hosted NIM containers (such as `nemotron-3-nano`), you may encounter a permission denied error during model manifest download, even when your API key is correct:

```
INFO 2026-02-02 12:52:12.892 nim_sdk.py:376] Downloading manifest profile: 2d9b8aac4a16d01e22e86db6b130e32889dcf73a2b28a996495c0904b9773453
ERROR 2026-02-02 12:52:13.083 nim_sdk.py:338] Download failed after 1 attempts. Last exception: I/O error Permission denied (os error 13)
ERROR 2026-02-02 12:52:13.083 nimutils.py:57] Error downloading models, ignoring and continuing startup
...
nimlib.exceptions.ManifestDownloadError: Error downloading manifest: I/O error Permission denied (os error 13)
```

This error typically occurs on systems where the user ID is not `1000` or `0`. Some NIM container expects specific user permissions for writing to the model cache directory.

**Solution:** Set `USERID=0` instead of `USERID=$(id -u)` when starting the NIM containers.

```bash
# Instead of:
export USERID=$(id -u)

# Use:
export USERID=0
```

Then restart your NIM containers:

```bash
docker compose -f deploy/compose/nims.yaml up -d nim-llm
```

:::{note}
This issue is specific to self-hosted NIM deployments when modifying the `nims.yaml` file to use different NIM images (e.g., `nemotron-3-nano`).
:::



## 429 Rate Limit Issue for NVIDIA-Hosted Models

You may encounter a "429 Client Error: Too Many Requests for url" error during ingestion when using NVIDIA-hosted models. This error indicates that the rate limiting threshold for the API has been exceeded. This is not an application issue, but rather a constraint imposed by the API service.

To mitigate rate limiting issues, configure the following parameters before starting ingestor-server and nv-ingest-ms-runtime:

```bash
export NV_INGEST_FILES_PER_BATCH=1
export NV_INGEST_CONCURRENT_BATCHES=1
export MAX_INGEST_PROCESS_WORKERS=1
export NV_INGEST_MAX_UTIL=8

# Start the ingestor-server and nv-ingest-ms-runtime containers
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
```

:::{note}
This can reduce the page-per-second performance for the ingestion. For maximum performance, [on-prem deployment](deploy-docker-self-hosted.md) is recommended.
:::

## Confidence threshold filtering issues

If no documents are returned when using confidence threshold filtering, the threshold may be set too high. Try lowering the `confidence_threshold` value or ensure the reranker is enabled to provide relevance scores. Confidence threshold filtering works best when reranker is enabled. Without reranker, documents may not have meaningful relevance scores. For optimal results, use confidence threshold values between 0.3-0.7. Values above 0.7 may be too restrictive.



## Deploy.Resources.Reservations.devices error

You might encounter an error resembling the following during the [container build process for self-hosted models](deploy-docker-self-hosted.md) process.
This is likely caused by an [outdated Docker Compose version](https://github.com/docker/compose/issues/11097).
To resolve this issue, upgrade Docker Compose to version `v2.29.0` or later.

```
1 error(s) decoding:

* error decoding 'Deploy.Resources.Reservations.devices[0]': invalid string value for 'count' (the only value allowed is 'all')
```



## Device error

You might encounter an `unknown device` error during the [container build process for self-hosted models](deploy-docker-self-hosted.md).
This error typically indicates that the container is attempting to access GPUs that are unavailable or non-existent on the host.
To resolve this issue, verify the GPU count specified in the [nims.yaml](../deploy/compose/nims.yaml) configuration file.

```bash
nvidia-container-cli: device error: {n}: unknown device: unknown
```



## DNS resolution failed for <service_name:port>
This category of errors in either `rag-server` or `ingestor-server` container logs indicates:
The server is trying to reach a self-hosted on-premises deployed service at `service_name:port` but it is unreachable. You can ensure that the service is up using `docker ps`.

For example, the below logs in ingestor server container indicates `page-elements` service is unreachable at port `8001`:

```output
Original error: Error during NimClient inference [yolox-page-elements, grpc]: [StatusCode.UNAVAILABLE] DNS resolution failed for page-elements:8001: C-ares status is not ARES_SUCCESS qtype=AAAA name=page-elements is_balancer=0: Could not contact DNS servers
```

In case you were expecting to use NVIDIA-hosted model for this service, then ensure the corresponding environment variables were set in the same terminal from where you did docker compose up. Following the above example the environment variables which are expected to be set are:

```output
   export YOLOX_HTTP_ENDPOINT="https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v3"
   export YOLOX_INFER_PROTOCOL="http"
```



## Elasticsearch connection timeout

If you encounter Elasticsearch connection timeout errors during ingestion, you can adjust the `ES_REQUEST_TIMEOUT` environment variable to increase the timeout duration. This is particularly useful when dealing with large documents or slow Elasticsearch clusters.

To resolve this issue on Helm deployments, do the following:

Add the `ES_REQUEST_TIMEOUT` environment variable to the `envVars` section in your `values.yaml` file:

```yaml
envVars:
  # ... existing environment variables ...
  ES_REQUEST_TIMEOUT: "1200"  # Timeout in seconds (default is typically 600)
```

To resolve this issue on Docker deployments, do the following:

Add the `ES_REQUEST_TIMEOUT` environment variable to the `environment` section in your `docker-compose-ingestor-server.yaml` file:

```yaml
environment:
  # ... existing environment variables ...
  ES_REQUEST_TIMEOUT: "1200"  # Timeout in seconds (default is typically 600)
```

After updating the configuration, restart the ingestor server and try the ingestion again. You can increase the timeout value if you continue to experience connection issues, but be aware that very high timeout values may indicate underlying performance issues with your Elasticsearch cluster.



## Embedding Model Dimensions Error

```
This model does not support 'dimensions', but a value of '2048' was provided.
```

This error occurs when using an embedding model with fixed output dimensions (e.g., `nvidia/nv-embedqa-e5-v5` with 1024 dimensions) while the default 2048 dimensions is configured. Some embedding models have fixed output dimensions and do not accept a `dimensions` parameter.

**Solution:** Configure the `APP_EMBEDDINGS_DIMENSIONS` environment variable or `embeddings.dimensions` in config.yaml to match your model's output dimensions:

```bash
export APP_EMBEDDINGS_DIMENSIONS=1024
export APP_EMBEDDINGS_MODELNAME='nvidia/nv-embedqa-e5-v5'
```

See [Configure Embedding Dimensions](change-model.md#configure-embedding-dimensions) for detailed instructions.

:::{warning}
If you change the embedding model or dimensions after ingesting documents, you must re-ingest your documents for accurate retrieval results.
:::



## Error details: [###] Too many open files for llama-3.3-nemotron-super-49b-v1.5 container
source: hyper_util::client::legacy::Error(Connect, ConnectError("dns error", Os { code: 24, kind: Uncategorized, message: "Too many open files" })) })

This error happens because the default number of Open files allowed are 1024 for Containers. Follow the below steps to modify the container configuration to allow more number of open files.

```sh
sudo mkdir -p /etc/systemd/system/containerd.service.d
echo "[Service]" | sudo tee /etc/systemd/system/containerd.service.d/override.conf
echo "LimitNOFILE=65536" | sudo tee -a /etc/systemd/system/containerd.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart containerd
sudo systemctl restart kubelet
```



## ERROR: pip's dependency resolver during container building

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behavior is the source of the following dependency conflicts.
```

If the above error related to dependency conflicts are seen while building containers, clear stale docker images using `docker system prune -af` and then execute the build command using `--no-cache` flag.



## External Vector databases

We've integrated VDB and embedding creation directly into the pipeline with caching included for expediency.
However, in a production environment, it's better to use a separately managed VDB service.

NVIDIA offers optimized models and tools like NVIDIA NeMo Retriever ([build.nvidia.com/explore/retrieval](https://build.nvidia.com/explore/retrieval))
and cuVS ([github.com/rapidsai/cuvs](https://github.com/rapidsai/cuvs)).



## Hallucination and Out-of-Context Responses

The current prompt configuration does not strictly enforce response generation from the retrieved context. This can result in the following scenarios:

1. **Out-of-context responses**: The LLM generates responses that are not grounded in the provided context
2. **Irrelevant context usage**: The model provides information from the retrieved context that doesn't directly answer the user's query

These issues can be addressed by adding the following instruction to the `rag_chain` user prompt in [prompt.yaml](../src/nvidia_rag/rag_server/prompt.yaml):

```yaml
Handling Missing Information: If the context does not contain the answer, you must state directly that you do not have information on the specific subject of the user's query. For example, if the query is about the "capital of France", your response should be "I did not find information about capital of France." Do not add any other words, apologies, or explanations.
```

:::{important}
Adding this information may impact response accuracy, especially when partial information is available instead of complete information in the retrieved context. The system may become more conservative in providing answers, potentially refusing to respond even when some relevant information exists in the context.
:::



## Helm Deployment Issues

### PVCs in Pending state (StorageClass issues)
If NIM Cache PVCs (e.g., `nemoretriever-embedding-ms-cache-pvc`) remain in `Pending` state, check if they are requesting a `storageClassName: default` that does not exist.
**Fix:** Ensure you have a default storage class. If using `local-path`, you can create an alias:
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: default
provisioner: rancher.io/local-path
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
```

### ProvisioningFailed (Access Mode mismatch)
If using `local-path` provisioner, it does not support `ReadWriteMany` access mode, which is the default for some NIM Caches.
**Fix:** Patch the NIMCache resources to use `ReadWriteOnce`:
```bash
kubectl patch nimcache nemoretriever-page-elements-v3 -n rag --type='merge' -p '{"spec":{"storage":{"pvc":{"volumeAccessMode":"ReadWriteOnce"}}}}'
# Repeat for other affected caches (table-structure-v1, ocr-v1, graphic-elements-v1)
kubectl delete pvc nemoretriever-page-elements-v3-pvc -n rag --wait=false # Delete pending PVC to trigger recreation
```



## Ingestion failures

In case a PDF or PPTx file is not ingested properly, check if that PDF/PPTx only contains images. If the images contain text that you want to extract, try enabling `APP_NVINGEST_EXTRACTINFOGRAPHICS` from [`deploy/compose/docker-compose-ingestor-server.yaml`](../deploy/compose/docker-compose-ingestor-server.yaml).

You may also enable image captioning to better extract content from images. For more details on enabling image captioning, refer to [image_captioning.md](image_captioning.md).



## IPv6-Only Computers

To use the NVIDIA RAG Blueprint with Docker on an IPv6-only computer, add the following code to your yaml file. For details, refer to [Use IPv6 networking](https://docs.docker.com/engine/daemon/ipv6/).

```yaml
networks:
 default:
  enable_ipv6: true
  name: nvidia-rag
```



## Node exporter pod crash with prometheus stack enabled in helm deployment

If you experience issues with the `prometheus-node-exporter` pod crashing after enabling the `kube-prometheus-stack`, and you encounter an error message like:

```sh
msg="listen tcp 0.0.0.0:9100: bind: address already in use"
```

This error indicates that the port `9100` is already in use. To resolve this, you can update the port for `prometheus-node-exporter` in the `values.yaml` file.

Update the following in `values.yaml`:

```yaml
kube-prometheus-stack:
   # ... existing code ...
  prometheus-node-exporter:
    service:
      port: 9101 # Changed from 9100 to 9101
      targetPort: 9101  # Changed from 9100 to 9101
```



## Out of memory issues while deploying nim-llm service

If you run into `torch.OutOfMemoryError: CUDA out of memory.` while deploying the model, this is most likely due to wrong model profile being auto selected during deployment. Refer to steps in the appropriate [deployment guide](readme.md#deployment-options-for-rag-blueprint) and set the correct profile using `NIM_MODEL_PROFILE` variable.



## Password Issue Fix

If you encounter any `password authentication failed` issues with the structured retriever container,
consider removing the volumes directory located at `deploy/compose/volumes`.
In this case, you may need to reprocess the data ingestion.



## pymilvus error: not allowed to retrieve raw data of field sparse

```
pymilvus.exceptions.MilvusException: <MilvusException: (code=65535, message=not allowed to retrieve raw data of field sparse)>
```
This happens when a collection created with vector search type `hybrid` is accessed using vector search type `dense` on retrieval side. Make sure both the search types are same in ingestor-server-compose and rag-server-compose file using `APP_VECTORSTORE_SEARCHTYPE` environment variable.



## Reset the entire cache

To reset the entire cache, you can run the following command.
This deletes all the volumes associated with the containers, including the cache.

```bash
docker compose down -v
```



## Running out of credits

If you run out of credits for the NVIDIA API Catalog,
you will need to obtain more credits to continue using the API.
Please contact your NVIDIA representative to get more credits.





## Related Topics

- [Debugging](debugging.md)
- [Release Notes](release-notes.md)

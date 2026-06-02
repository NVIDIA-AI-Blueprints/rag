<!--
  SPDX-FileCopyrightText: Copyright (c) 2025, 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Deploy NVIDIA RAG Blueprint on OpenShift with Helm

Use the following documentation to deploy the [NVIDIA RAG Blueprint](readme.md) on a Red Hat OpenShift cluster by using Helm.

- To deploy on standard Kubernetes (non-OpenShift), refer to [Deploy on Kubernetes with Helm](deploy-helm.md).
- To deploy with MIG support, refer to [RAG Deployment with MIG Support](mig-deployment.md).
- For other deployment options, refer to [Deployment Options](readme.md#deployment-options-for-rag-blueprint).

The chart includes built-in OpenShift support gated behind an `openshift.enabled` flag.
When enabled, the chart automatically creates OpenShift Routes with edge TLS and an `anyuid` SCC RoleBinding for all required ServiceAccounts — no manual `oc adm policy` commands are needed.


## Prerequisites

:::{important}
Ensure you have at least 200GB of available disk space per node where NIMs will be deployed. This space is required for the following:
- NIM model cache downloads (~100-150GB)
- Container images (~20-30GB)
- Persistent volumes for vector database and application data
- Logs and temporary files
:::

1. [Get an API Key](api-key.md).

2. Verify that you meet the [hardware requirements](support-matrix.md). The minimum GPU requirements depend on deployment mode:

   | Deployment Mode | GPUs Required | Notes |
   |----------------|--------------|-------|
   | Full (self-hosted NIMs) | 8–10 | All NIM models running in-cluster |
   | Minimal (no VLM, no optional NIMs) | 6–7 | Core pipeline without VLM or audio |
   | API-hosted LLM | 4–6 | LLM via [build.nvidia.com](https://build.nvidia.com/); self-hosted embedding, reranking, and NV-Ingest NIMs |

3. Verify that you have **OpenShift 4.14 or later** with cluster-admin access, and the `oc` CLI configured.

4. Verify that you have **Helm 3** installed. To install Helm 3, follow the official [Helm installation instructions](https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3).

5. Verify that you have the **NVIDIA GPU Operator** installed and functional. For details, see [GPU Operator documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html).

6. Verify that you have the **NVIDIA NIM Operator** v3.0.2+ installed. If not, install it:

    ```sh
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
      --username='$oauthtoken' \
      --password=$NGC_API_KEY
    helm repo update
    helm install nim-operator nvidia/k8s-nim-operator -n nim-operator --create-namespace
    ```

    For details, see [NIM Operator installation guide](https://docs.nvidia.com/nim-operator/latest/install.html).

7. Install the **Elastic Cloud on Kubernetes (ECK) operator**. Elasticsearch is the default vector database for this chart, and the chart provisions an Elasticsearch CR that requires the ECK operator to reconcile it:

    ```sh
    helm repo add elastic https://helm.elastic.co
    helm repo update
    helm install elastic-operator elastic/eck-operator -n elastic-system --create-namespace
    ```

    If you plan to replace Elasticsearch with Milvus or another backend and disable the chart-managed Elasticsearch, skip this step. See [Vector database configuration](change-vectordb.md).

8. Verify that a **default StorageClass** with dynamic provisioning is available (e.g., `gp3-csi` on AWS):

    ```sh
    oc get storageclass
    ```

    :::{note}
    If your cluster does not have a default dynamic StorageClass available (common on bare-metal OpenShift installations), install the OpenEBS Dynamic LocalPV Provisioner to satisfy the chart's PVC requirements:

    ```sh
    # Add the OpenEBS Helm repository
    helm repo add openebs https://openebs.github.io/openebs
    helm repo update

    # Create the openebs namespace
    kubectl create namespace openebs

    # Install only the LocalPV provisioner; disable other storage engines
    # On OpenShift, also disable the bundled minio/loki/alloy subcharts —
    # their pods violate the restricted PodSecurity policy and the
    # `openebs-minio-post-job` fails with BackoffLimitExceeded otherwise.
    helm install openebs openebs/openebs \
      --namespace openebs \
      --set engines.replicated.mayastor.enabled=false \
      --set engines.local.lvm.enabled=false \
      --set engines.local.zfs.enabled=false \
      --set minio.enabled=false \
      --set loki.enabled=false \
      --set alloy.enabled=false

    # OpenShift requires the privileged SCC for the provisioner service account
    oc adm policy add-scc-to-user privileged -z openebs-localpv-provisioner -n openebs

    # Mark openebs-hostpath as the default StorageClass
    kubectl patch storageclass openebs-hostpath \
      -p '{"metadata": {"annotations": {"storageclass.kubernetes.io/is-default-class": "true"}}}'
    ```

    Verify that the provisioner pods are running and the StorageClass is configured as default:

    ```sh
    kubectl get pods -n openebs
    kubectl get sc
    ```
    :::

9. Check GPU node taints. GPU nodes on OpenShift clusters typically have taints that prevent non-GPU workloads from scheduling on them. You need the taint keys for the tolerations configuration:

    ```sh
    oc get nodes -l nvidia.com/gpu.present=true \
      -o custom-columns="NODE:.metadata.name,TAINTS:.spec.taints[*].key"
    ```

10. Verify the kubelet `podPidsLimit` is at least `16384`. The `rag-nv-ingest` pod, along with the reranker and other NIMs, collectively spawn several thousand threads at steady state. The OpenShift default of `4096` is insufficient and surfaces as `pthread_create failed: Resource temporarily unavailable` errors during ingestion and reranking.

    Inspect the current value on any worker node:

    ```sh
    oc get --raw /api/v1/nodes/<node-name>/proxy/configz \
      | jq '.kubeletconfig.podPidsLimit'
    ```

    If the value is below `16384`, apply the following `KubeletConfig` (cluster-admin access required). The Machine Config Operator will roll the affected nodes:

    ```yaml
    apiVersion: machineconfiguration.openshift.io/v1
    kind: KubeletConfig
    metadata:
      name: rag-pod-pids-limit
    spec:
      machineConfigPoolSelector:
        matchLabels:
          pools.operator.machineconfiguration.openshift.io/worker: ""
      kubeletConfig:
        podPidsLimit: 16384
    ```

11. Accept NIM licenses. Each NIM container image on NGC requires individually accepting a license agreement before your API key can pull it. Accept licenses for each NIM at [build.nvidia.com](https://build.nvidia.com/).


## Deploy the RAG Helm Chart

:::{important}
When you use the Helm NIM Operator deployment, approximately 60 to 70 minutes is required for the entire pipeline to reach a running state on first deploy. Subsequent deployments are significantly faster (~10-15 minutes) because model caches are already populated.
:::

To deploy the RAG Blueprint on OpenShift, use the following procedure.

1. Set your environment variables.

    ```sh
    export NGC_API_KEY="nvapi-..."
    export NAMESPACE="rag"
    ```

2. Navigate to the chart directory and build dependencies.

    ```sh
    cd deploy/helm/nvidia-blueprint-rag

    helm repo add nvidia-nemo https://helm.ngc.nvidia.com/nvidia/nemo-microservices \
      --username '$oauthtoken' --password "$NGC_API_KEY"

    helm dependency build
    ```

    :::{note}
    The OpenShift overlay passes values through the `nv-ingest` subchart's
    `extraVolumes` / `extraVolumeMounts` keys. With the currently pinned
    `nv-ingest` 26.3.0, those values need a small indent adjustment in the
    pulled chart before `helm upgrade` will render valid YAML. Re-apply this
    after every `helm dependency build` or `helm dependency update`:

    ```sh
    mkdir -p /tmp/nvi && \
      tar xzf charts/nv-ingest-26.3.0.tgz -C /tmp/nvi && \
      sed -i '/toYaml $v | nindent 12/s/nindent 12/nindent 14/' \
        /tmp/nvi/nv-ingest/templates/deployment.yaml && \
      tar czf charts/nv-ingest-26.3.0.tgz -C /tmp/nvi nv-ingest && \
      rm -rf /tmp/nvi
    ```
    :::

    :::{note}
    **Alternative — installing from the NGC chart URL**

    If you prefer the install-from-NGC pattern shown in [Deploy on Kubernetes with Helm](deploy-helm.md) instead of cloning this repo, pull the chart locally first. Helm cannot patch a chart it streams directly from a remote URL, so the indent adjustment must be applied to a local copy before install:

    ```sh
    # Pull and untar the chart from NGC. The NGC package ships with the
    # nv-ingest subchart already extracted under charts/nv-ingest/, so the
    # sed below can edit the template file in place.
    helm pull https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.6.0.tgz \
      --username '$oauthtoken' --password "$NGC_API_KEY" \
      --untar --untardir /tmp

    # Apply the indent adjustment to the bundled nv-ingest subchart
    sed -i '/toYaml $v | nindent 12/s/nindent 12/nindent 14/' \
      /tmp/nvidia-blueprint-rag/charts/nv-ingest/templates/deployment.yaml

    # Install from the patched local directory
    helm upgrade --install rag -n $NAMESPACE /tmp/nvidia-blueprint-rag \
      -f /tmp/nvidia-blueprint-rag/values-openshift.yaml \
      --set imagePullSecret.password="$NGC_API_KEY" \
      --set ngcApiSecret.password="$NGC_API_KEY" \
      --timeout 15m
    ```

    This replaces steps 2 and 4 of the procedure above; steps 3 and 5 are unchanged.
    :::

3. Create a namespace.

    ```sh
    oc new-project $NAMESPACE
    ```

4. Install the Helm chart with the OpenShift overlay.

    ```sh
    helm upgrade --install rag -n $NAMESPACE . \
      -f values-openshift.yaml \
      --set imagePullSecret.password="$NGC_API_KEY" \
      --set ngcApiSecret.password="$NGC_API_KEY" \
      --timeout 15m
    ```

    The `values-openshift.yaml` overlay enables the following:
    - **OpenShift Routes** for the frontend and RAG server with edge TLS
    - **anyuid SCC RoleBinding** for all ServiceAccounts that need it
    - **ClusterIP** service type for the frontend (Routes handle external access)

    :::{note}
    If your GPU nodes have taints, you must add tolerations. Pass them on the command line with `--set-json` or create a values overlay file.
    For example, if your GPU nodes have a `gpu-taint` taint:

    ```sh
    helm upgrade --install rag -n $NAMESPACE . \
      -f values-openshift.yaml \
      --set imagePullSecret.password="$NGC_API_KEY" \
      --set ngcApiSecret.password="$NGC_API_KEY" \
      --set-json 'nimOperator.nim-llm.tolerations=[{"key":"gpu-taint","operator":"Exists","effect":"NoSchedule"}]' \
      --set-json 'nimOperator.nvidia-nim-llama-nemotron-embed-1b-v2.tolerations=[{"key":"gpu-taint","operator":"Exists","effect":"NoSchedule"}]' \
      --set-json 'nimOperator.nvidia-nim-llama-nemotron-rerank-1b-v2.tolerations=[{"key":"gpu-taint","operator":"Exists","effect":"NoSchedule"}]' \
      --set-json 'nv-ingest.nimOperator.ocr.tolerations=[{"key":"gpu-taint","operator":"Exists","effect":"NoSchedule"}]' \
      --set-json 'nv-ingest.nimOperator.page_elements.tolerations=[{"key":"gpu-taint","operator":"Exists","effect":"NoSchedule"}]' \
      --timeout 15m
    ```

    The chart also includes a `values-openshift-test.yaml` reference overlay that demonstrates tolerations, resource tuning, disabled observability, and API-hosted LLM mode. Edit the toleration keys to match your cluster and layer it on with `-f values-openshift-test.yaml`.
    :::

5. Link the NGC pull secret to the NIM Operator ServiceAccount.

    The NIM Operator creates a `nim-cache-sa` ServiceAccount for model cache jobs. Link the pull secret so it can pull NIM model images:

    ```sh
    oc secrets link nim-cache-sa ngc-secret --for=pull -n $NAMESPACE
    ```

    If NIMCache pods are stuck in `ImagePullBackOff`, delete them so the operator recreates them with the linked secret:

    ```sh
    oc delete pod -l app.nvidia.com/nim-cache -n $NAMESPACE
    ```


## Verify a Deployment

1. List the pods by running the following code.

    ```sh
    oc get pods -n $NAMESPACE
    ```

    You should see output similar to the following.

    ```sh
    NAME                                          READY   STATUS    AGE
    ingestor-server-xxxxxxxxx-xxxxx               1/1     Running   5m
    rag-eck-elasticsearch-es-default-0            1/1     Running   5m
    nemotron-embedding-ms-xxxxxxxxx-xxxxx         1/1     Running   10m
    nemotron-graphic-elements-v1-xxxxxxxxx-xxxxx  1/1     Running   10m
    nemotron-ocr-v1-xxxxxxxxx-xxxxx               1/1     Running   10m
    nemotron-page-elements-v3-xxxxxxxxx-xxxxx     1/1     Running   10m
    nemotron-ranking-ms-xxxxxxxxx-xxxxx           1/1     Running   10m
    nemotron-table-structure-v1-xxxxxxxxx-xxxxx   1/1     Running   10m
    nim-llm-xxxxxxxxx-xxxxx                       1/1     Running   15m
    rag-frontend-xxxxxxxxx-xxxxx                  1/1     Running   5m
    rag-nv-ingest-xxxxxxxxx-xxxxx                 1/1     Running   5m
    rag-redis-master-0                            1/1     Running   5m
    rag-redis-replicas-0                          1/1     Running   5m
    rag-seaweedfs-all-in-one-xxxxxxxxx-xxxxx      1/1     Running   5m
    rag-server-xxxxxxxxx-xxxxx                    1/1     Running   5m
    ```

    If you have enabled Milvus instead of the default Elasticsearch vector database (see [Vector database configuration](change-vectordb.md)), the list also includes `rag-etcd-0` and `rag-minio-xxx` pods.

   :::{note}
   Model downloads do not show detailed progress indicators in pod status. Pods may appear in "ContainerCreating" or "Init" state for extended periods while models download in the background.

   You can monitor the deployment progress by running the following code.

   ```sh
   # Check NIMCache download status (shows if cache is ready)
   oc get nimcache -n $NAMESPACE

   # Check NIMService status
   oc get nimservice -n $NAMESPACE

   # Check events for detailed information
   oc get events -n $NAMESPACE --sort-by='.lastTimestamp'

   # Watch logs of a specific pod to see detailed progress
   oc logs -f <pod-name> -n $NAMESPACE
   ```
   :::

2. Verify OpenShift Routes are created.

    ```sh
    oc get routes -n $NAMESPACE
    ```

3. Get the application URLs.

    ```sh
    # Frontend URL
    echo "https://$(oc get route rag-frontend -n $NAMESPACE -o jsonpath='{.spec.host}')"

    # API URL
    echo "https://$(oc get route rag-server -n $NAMESPACE -o jsonpath='{.spec.host}')"

    # API health check
    API_HOST=$(oc get route rag-server -n $NAMESPACE -o jsonpath='{.spec.host}')
    curl -sk "https://${API_HOST}/health"
    ```


## Experiment with the Web User Interface

Open a web browser and access the frontend URL from the previous step. You can start experimenting by uploading documents and asking questions. For details, see [User Interface for NVIDIA RAG Blueprint](user-interface.md).

:::{note}
Unlike standard Kubernetes deployments, OpenShift Routes provide external access directly — no `kubectl port-forward` is needed.
:::


## Using NVIDIA-Hosted Models (Reduced GPU Requirements)

For clusters with limited GPU capacity, you can use NVIDIA-hosted model endpoints at [build.nvidia.com](https://build.nvidia.com/) for the LLM while keeping embedding, reranking, and NV-Ingest NIMs self-hosted.

Set the LLM server URLs to empty strings and disable the self-hosted NIM LLM:

```yaml
nimOperator:
  nim-llm:
    enabled: false

envVars:
  APP_LLM_SERVERURL: ""
  APP_QUERYREWRITER_SERVERURL: ""
  APP_FILTEREXPRESSIONGENERATOR_SERVERURL: ""
  REFLECTION_LLM_SERVERURL: ""

ingestor-server:
  envVars:
    SUMMARY_LLM_SERVERURL: ""
```

The included `values-openshift-test.yaml` overlay implements this pattern. Layer it on with `-f values-openshift-test.yaml`.


## Change a Deployment

To change an existing deployment, after you modify the values files, run the following code.

```sh
helm upgrade rag -n $NAMESPACE . \
  -f values-openshift.yaml \
  --set imagePullSecret.password="$NGC_API_KEY" \
  --set ngcApiSecret.password="$NGC_API_KEY"
```


## Uninstall a Deployment

To uninstall a deployment, run the following code.

```sh
helm uninstall rag -n $NAMESPACE
```

Run the following code to remove the NIMCache and Persistent Volume Claims (PVCs) created by the chart which are not removed by default.

```sh
oc delete nimcache --all -n $NAMESPACE
oc delete nimservice --all -n $NAMESPACE
oc delete pvc --all -n $NAMESPACE
```

To delete the namespace entirely:

```sh
oc delete namespace $NAMESPACE
```


## OpenShift-Specific Troubleshooting

### Security Context Constraints (SCC)

**Symptom**: Pods fail with `CrashLoopBackOff` and logs show permission errors such as `mkdir: cannot create directory '/opt/nim/.cache': Permission denied`.

**Why**: OpenShift's default `restricted` SCC assigns random UIDs. NIM containers and infrastructure services expect to run as specific users.

**Fix**: The chart's `openshift.yaml` template automatically grants the `anyuid` SCC to required ServiceAccounts when `openshift.enabled` is `true`. If you are not using `values-openshift.yaml`, grant `anyuid` manually:

```sh
oc adm policy add-scc-to-user anyuid -z default -n $NAMESPACE
```

### nv-ingest Ray Worker Failures on Clusters with Low `podPidsLimit`

**Symptom**: The `rag-nv-ingest` pod restarts repeatedly with `pthread_create failed: Resource temporarily unavailable` in its logs. Ingestion tasks remain in the `pending` state and the Redis queue (`LLEN ingest_task_queue`) does not drain.

**Why**: The pod's cgroup `cpuset.cpus` reflects the full host CPU set (for example, `0-255`), so Ray detects all host CPUs and prestarts an equally large Python worker pool. Each worker spawns several gRPC threads during initialization. On clusters where the kubelet enforces the default `podPidsLimit` of 4096, the cumulative thread count exceeds the cgroup's PID ceiling, and worker processes are terminated before they can register with the raylet.

**Recommended fix**: Raise the kubelet `podPidsLimit` to `16384` via a `KubeletConfig` custom resource. See Prerequisites step 10 for the manifest. This is the cluster-level change that addresses the root cause.

**Workaround when the cluster `podPidsLimit` cannot be raised**: The `values-openshift.yaml` overlay enables a `sitecustomize.py` ConfigMap (`nv-ingest.pyPatches.enabled: true`) that overrides `os.cpu_count` and `psutil.cpu_count` to return the value of `RAG_NV_INGEST_DETECTED_CPUS` (default `4`). The overlay also sets `MAX_INGEST_PROCESS_WORKERS=4` to cap the number of Ray actor replicas per pipeline stage. Together, these settings keep the pod's steady-state PID count well below the cgroup limit at the cost of slower per-document throughput.

**Tuning the worker count**: To change the worker count, update the values in `values-openshift.yaml` and re-run `helm upgrade`:

```yaml
nv-ingest:
  envVars:
    RAG_NV_INGEST_DETECTED_CPUS: "8"   # increase to improve throughput
    MAX_INGEST_PROCESS_WORKERS: "8"    # keep aligned with the value above
```

Alternatively, override the values on the command line without editing the file:

```sh
helm upgrade --install rag -n $NAMESPACE . \
  -f values-openshift.yaml \
  --set 'nv-ingest.envVars.RAG_NV_INGEST_DETECTED_CPUS=8' \
  --set 'nv-ingest.envVars.MAX_INGEST_PROCESS_WORKERS=8' \
  --set imagePullSecret.password="$NGC_API_KEY" \
  --set ngcApiSecret.password="$NGC_API_KEY"
```

Higher values reduce per-document ingestion latency but increase the pod's PID consumption. Values above `8` are not recommended unless the kubelet `podPidsLimit` has first been raised (typically to `16384`) via the `KubeletConfig` manifest in Prerequisites step 10.

### Reranker HTTP 500 Errors from Thread Pool Initialization Failure

**Symptom**: The `rag-server` logs report `[500] Unknown Error` during query generation. The `nemotron-ranking-ms` pod logs contain `ThreadPoolBuildError { kind: IOError(Os { code: 11, kind: WouldBlock }) }` originating in the HuggingFace tokenizer path.

**Why**: The reranker NIM's Rust/Rayon thread pool defaults to one thread per host CPU. On nodes that expose the full host cpuset, initialization exceeds the kubelet `podPidsLimit` and the NIM returns HTTP 500. The base chart sets thread caps on the OCR and YOLOX NIMs but not on the reranker.

**Recommended fix**: Raise the kubelet `podPidsLimit` to `16384` via a `KubeletConfig` custom resource. See Prerequisites step 10 for the manifest.

**Workaround when the cluster `podPidsLimit` cannot be raised**: The `values-openshift.yaml` overlay sets `RAYON_NUM_THREADS=4` and `TOKENIZERS_PARALLELISM=false` on the reranker NIM. To adjust the cap, edit the value in `values-openshift.yaml` and re-run `helm upgrade`.

### GPU Node Scheduling and Tolerations

**Symptom**: NIM pods stay in `Pending` state.

**Why**: GPU nodes typically have taints. NIM workloads need matching tolerations.

**Fix**: Discover your taint keys and set tolerations in your values file:

```sh
oc get nodes -l nvidia.com/gpu.present=true \
  -o custom-columns="NODE:.metadata.name,TAINTS:.spec.taints[*].key"
```

Set matching tolerations for each NIM component via `--set-json` or a values overlay. The `values-openshift-test.yaml` file demonstrates the pattern.

### NIM LLM VRAM Requirements

**Symptom**: NIM LLM pod crashes during model loading with `torch.OutOfMemoryError`.

**Fix**: For GPUs with limited VRAM, reduce `NIM_MAX_MODEL_LEN` or use NVIDIA-hosted models as described in [Using NVIDIA-Hosted Models](#using-nvidia-hosted-models-reduced-gpu-requirements).

### Route Timeouts

**Symptom**: Document ingestion or complex queries return `504 Gateway Timeout`.

**Why**: OpenShift's default Route timeout is 30 seconds. The chart sets `haproxy.router.openshift.io/timeout: 300s` on the RAG server Route, but if you create Routes manually, set this annotation explicitly.

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Pods stuck in `Pending` | Missing tolerations or insufficient GPU resources | Check taints; set tolerations in values |
| `ImagePullBackOff` | Missing NGC secret or unaccepted NIM license | Verify `ngc-secret` exists; accept licenses at [build.nvidia.com](https://build.nvidia.com/) |
| `CrashLoopBackOff` | SCC restrictions or insufficient memory | Enable `openshift.enabled`; check resource limits |
| NIM LLM `OOMKilled` | Insufficient VRAM | Reduce `NIM_MAX_MODEL_LEN` or use NVIDIA-hosted LLM |
| PVC `Pending` | StorageClass not found | Set correct `storageClass` in values or use `""` for default |
| `504 Gateway Timeout` | Route timeout too low | Annotate route with `haproxy.router.openshift.io/timeout=300s` |
| NIMCache `ImagePullBackOff` | Pull secret not linked to `nim-cache-sa` | Run `oc secrets link nim-cache-sa ngc-secret --for=pull` |
| Ingest tasks stuck `pending` | nv-ingest Ray workers hit `podPidsLimit` | See [nv-ingest Ray Worker Failures](#nv-ingest-ray-worker-failures-on-clusters-with-low-podpidslimit) |
| Reranker returns HTTP 500 with `ThreadPoolBuildError` | Rust/Rayon thread pool exceeds pod PID limit | See [Reranker HTTP 500 Errors](#reranker-http-500-errors-from-thread-pool-initialization-failure) |
| `helm upgrade` fails with `yaml: ... did not find expected '-' indicator` | Indent adjustment needed in pulled `nv-ingest` 26.3.0 chart | Re-apply the post-`dependency build` step in [Deploy step 2](#deploy-the-rag-helm-chart) |


## Troubleshooting Helm Issues

For general troubleshooting issues with Helm deployment, refer to [Troubleshooting](troubleshooting.md).


## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Deploy on Kubernetes with Helm](deploy-helm.md)
- [Best Practices for Common Settings](accuracy_perf.md)
- [User Interface](user-interface.md)
- [Troubleshoot](troubleshooting.md)

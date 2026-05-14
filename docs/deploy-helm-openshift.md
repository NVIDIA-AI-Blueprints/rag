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

7. Verify that a **default StorageClass** with dynamic provisioning is available (e.g., `gp3-csi` on AWS):

    ```sh
    oc get storageclass
    ```

8. Check GPU node taints. GPU nodes on OpenShift clusters typically have taints that prevent non-GPU workloads from scheduling on them. You need the taint keys for the tolerations configuration:

    ```sh
    oc get nodes -l nvidia.com/gpu.present=true \
      -o custom-columns="NODE:.metadata.name,TAINTS:.spec.taints[*].key"
    ```

9. Accept NIM licenses. Each NIM container image on NGC requires individually accepting a license agreement before your API key can pull it. Accept licenses for each NIM at [build.nvidia.com](https://build.nvidia.com/).


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
    nemotron-ocr-v1-xxxxxxxxx-xxxxx               1/1     Running   10m
    nemotron-page-elements-v3-xxxxxxxxx-xxxxx     1/1     Running   10m
    nemotron-ranking-ms-xxxxxxxxx-xxxxx           1/1     Running   10m
    nim-llm-xxxxxxxxx-xxxxx                       1/1     Running   15m
    rag-etcd-0                                    1/1     Running   5m
    rag-frontend-xxxxxxxxx-xxxxx                  1/1     Running   5m
    rag-minio-xxxxxxxxx-xxxxx                     1/1     Running   5m
    rag-nv-ingest-xxxxxxxxx-xxxxx                 1/1     Running   5m
    rag-redis-master-0                            1/1     Running   5m
    rag-server-xxxxxxxxx-xxxxx                    1/1     Running   5m
    ```

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


## Troubleshooting Helm Issues

For general troubleshooting issues with Helm deployment, refer to [Troubleshooting](troubleshooting.md).


## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Deploy on Kubernetes with Helm](deploy-helm.md)
- [Best Practices for Common Settings](accuracy_perf.md)
- [User Interface](user-interface.md)
- [Troubleshoot](troubleshooting.md)

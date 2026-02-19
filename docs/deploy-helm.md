<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Deploy NVIDIA RAG Blueprint on Kubernetes with Helm

Use this guide to deploy the [NVIDIA RAG Blueprint](readme.md) on a Kubernetes cluster with Helm.

- For MIG support, see [RAG Deployment with MIG Support](mig-deployment.md).
- To deploy from the repository, see [Deploy Helm from the Repository](deploy-helm-from-repo.md).
- For other options, see [Deployment Options](readme.md#deployment-options-for-rag-blueprint).

The deployment installs these core services:

- RAG server
- Ingestor server
- NV-Ingest


## Prerequisites

:::{important}
Ensure you have at least 200GB of available disk space per node where NIMs will be deployed. This space is required for the following:
- NIM model cache downloads (~100-150GB)
- Container images (~20-30GB)
- Persistent volumes for vector database and application data
- Logs and temporary files

Plan for more space if you enable persistence for multiple services.
:::

1. [Get an API Key](api-key.md).

2. Ensure you meet the [hardware requirements](support-matrix.md).

3. Ensure the NGC CLI is available on your client. Download it from [NGC CLI installers](https://ngc.nvidia.com/setup/installers/cli).

4. Ensure a supported Kubernetes version (for example, v1.28 or later) is installed and running on Ubuntu 22.04 or 24.04. See [Kubernetes documentation](https://kubernetes.io/docs/setup/) and [NVIDIA Cloud Native Stack](https://github.com/NVIDIA/cloud-native-stack).

5. Install Helm 3 (not Helm 4). Follow the [Helm v3 installation](https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3) instructions for your platform (for example, the `get-helm-3` script).

6. Ensure a default storage class exists for PVC provisioning. One option is the [Rancher local path provisioner](https://github.com/rancher/local-path-provisioner?tab=readme-ov-file#installation).

    ```console
    kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.26/deploy/local-path-storage.yaml
    kubectl get pods -n local-path-storage
    kubectl get storageclass
    ```

7. If the local path storage class is not the default, set it with the following command.

    ```sh
    kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
    ```

8. Install the [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) if you have not already.

9. (Optional) Enable [time slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html) to share GPUs between pods.

10. Install the NVIDIA NIM Operator if needed. Run the following commands:

    ```sh
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
      --username='$oauthtoken' \
      --password=$NGC_API_KEY
    helm repo update
    helm install nim-operator nvidia/k8s-nim-operator -n nim-operator --create-namespace
    ```

    For details, see [NIM Operator installation](https://docs.nvidia.com/nim-operator/latest/install.html).


## Deploy the RAG Helm Chart

:::{important}
With the Helm NIM Operator deployment, the full pipeline takes about 60–70 minutes to reach a running state.
:::

Use the following procedure to deploy the RAG server and Ingestor server.

Ensure your NGC API key is set in the environment (for example, `export NGC_API_KEY="<your-key>"`). See [Get an API Key](api-key.md).

1. Create the deployment namespace:

    ```sh
    kubectl create namespace rag
    ```

2. Install the Helm chart:

    ```sh
    helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.4.0.tgz \
    --username '$oauthtoken' \
    --password "${NGC_API_KEY}" \
    --set imagePullSecret.password=$NGC_API_KEY \
    --set ngcApiSecret.password=$NGC_API_KEY
    ```

   :::{important}
   **NVIDIA RTX 6000 Pro deployments**
   
    For NVIDIA RTX 6000 Pro GPUs (instead of H100), configure the NIM LLM model profile. The required configuration is in the [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) file but commented out.

    Uncomment and adjust the following under `nimOperator.nim-llm.model`:
    ```yaml
    model:
      engine: tensorrt_llm
      precision: "fp8"
      qosProfile: "throughput"
      tensorParallelism: "1"
      gpus:
        - product: "rtx6000_blackwell_sv"
    ```
   
   Then install with the modified values file:
   ```sh
   helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.4.0.tgz \
     --username '$oauthtoken' \
     --password "${NGC_API_KEY}" \
     --set imagePullSecret.password=$NGC_API_KEY \
     --set ngcApiSecret.password=$NGC_API_KEY \
     -f deploy/helm/nvidia-blueprint-rag/values.yaml
   ```
   :::

   :::{note}
   For non-default NIM LLM profiles, see [NIM Model Profile Configuration](model-profiles.md).
   :::


## Verify the Deployment

Use the following procedure to verify the deployment.

1. List the pods:

    ```sh
    kubectl get pods -n rag
    ```

    You should see output similar to the following.

   :::{note}
   If some pods stay in `Pending`, see [PVCs in Pending state (StorageClass issues)](troubleshooting.md#pvcs-in-pending-state-storageclass-issues).
   :::

    ```sh
    NAME                                                 READY   STATUS      RESTARTS   AGE
    ingestor-server-6cc886bcdf-6rfwm                     1/1     Running     0          54m
    milvus-standalone-7dd5db4755-ctqzg                   1/1     Running     0          54m
    nemoretriever-embedding-ms-86f75c8f65-dfhd2          1/1     Running     0          39m
    nemoretriever-graphic-elements-v1-67d9d65bdc-ftbkw   1/1     Running     0          33m
    nemoretriever-ocr-v1-78f56cddb9-f4852                1/1     Running     0          40m
    nemoretriever-page-elements-v3-56ddcf9b4b-qsg82      1/1     Running     0          49m
    nemoretriever-ranking-ms-5ff774889f-fwrlm            1/1     Running     0          40m
    nemoretriever-table-structure-v1-696c9f5665-l9sxn    1/1     Running     0          37m
    nim-llm-7cb9bdcc89-hwpkq                             1/1     Running     0          11m
    nim-llm-cache-job-77hpc                              0/1     Completed   0          94s
    rag-etcd-0                                           1/1     Running     0          54m
    rag-frontend-5db7874b77-49q8f                        1/1     Running     0          54m
    rag-minio-649f6476c-n29b8                            1/1     Running     0          54m
    rag-nv-ingest-6bf4d98866-kbgg7                       1/1     Running     0          54m
    rag-redis-master-0                                   1/1     Running     0          54m
    rag-redis-replicas-0                                 1/1     Running     0          54m
    rag-server-6d9cd4c677-ntzgz                          1/1     Running     0          54m
    ```

   :::{note}
   The full pipeline takes about 60–70 minutes to reach a running state. Time is spent on:

   - NIM model cache downloads (~40–50 minutes)
   - NIMService initialization (~10–15 minutes)
   - Pod startup and readiness (~5–10 minutes)

   Model downloads do not show progress in pod status. Pods can stay in `ContainerCreating` or `Init` while models download.

   To monitor progress, run:

   ```sh
   # Check pod status
   kubectl get pods -n rag

   # Check NIMCache download status (shows if cache is ready)
   kubectl get nimcache -n rag

   # Check NIMService status
   kubectl get nimservice -n rag

   # Check events for detailed information
   kubectl get events -n rag --sort-by='.lastTimestamp'

   # Watch logs of a specific pod to see detailed progress
   kubectl logs -f <pod-name> -n rag
   
   # Check PVC usage to monitor cache download size
   kubectl get pvc -n rag
   ```
   
   Later deployments are faster (~10–15 minutes) because caches are already populated.
   :::

2. List the services:

    ```sh
    kubectl get svc -n rag
    ```

    You should see output similar to the following.

    ```sh
    NAME                                TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)              AGE
    ingestor-server                     ClusterIP   10.107.12.217    <none>        8082/TCP             54m
    milvus                              ClusterIP   10.99.110.203    <none>        19530/TCP,9091/TCP   54m
    nemoretriever-embedding-ms          ClusterIP   10.104.99.15     <none>        8000/TCP,8001/TCP    54m
    nemoretriever-graphic-elements-v1   ClusterIP   10.96.115.45     <none>        8000/TCP,8001/TCP    54m
    nemoretriever-ocr-v1                ClusterIP   10.100.107.215   <none>        8000/TCP,8001/TCP    54m
    nemoretriever-page-elements-v3      ClusterIP   10.102.237.196   <none>        8000/TCP,8001/TCP    54m
    nemoretriever-ranking-ms            ClusterIP   10.96.114.244    <none>        8000/TCP,8001/TCP    54m
    nemoretriever-table-structure-v1    ClusterIP   10.107.227.139   <none>        8000/TCP,8001/TCP    54m
    nim-llm                             ClusterIP   10.104.60.155    <none>        8000/TCP,8001/TCP    54m
    rag-etcd                            ClusterIP   10.104.74.116    <none>        2379/TCP,2380/TCP    54m
    rag-etcd-headless                   ClusterIP   None             <none>        2379/TCP,2380/TCP    54m
    rag-frontend                        NodePort    10.100.190.142   <none>        3000:31473/TCP       54m
    rag-minio                           ClusterIP   10.101.18.143    <none>        9000/TCP             54m
    rag-nv-ingest                       ClusterIP   10.107.186.4     <none>        7670/TCP             54m
    rag-redis-headless                  ClusterIP   None             <none>        6379/TCP             54m
    rag-redis-master                    ClusterIP   10.105.178.202   <none>        6379/TCP             54m
    rag-redis-replicas                  ClusterIP   10.97.29.199     <none>        6379/TCP             54m
    rag-server                          ClusterIP   10.99.216.173    <none>        8081/TCP             54m
    ```


## Port-Forward to Access the Web UI

To reach the [RAG UI](user-interface.md) from your machine, run:

  ```sh
  kubectl port-forward -n rag service/rag-frontend 3000:3000 --address 0.0.0.0
  ```

Then open `http://localhost:3000` in a browser.

:::{note}
Port-forwarding is a quick way to try the UI. Large or bulk ingestion through the UI may hit timeouts.
::: 

## Experiment with the Web UI

Open the RAG UI in a browser. Upload documents and ask questions to try it out. For details, see [User Interface for NVIDIA RAG Blueprint](user-interface.md).


## Change a Deployment

After modifying [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), run the following from the repository root:

```sh
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.4.0.tgz \
--username '$oauthtoken' \
--password "${NGC_API_KEY}" \
--set imagePullSecret.password=$NGC_API_KEY \
--set ngcApiSecret.password=$NGC_API_KEY \
-f deploy/helm/nvidia-blueprint-rag/values.yaml
```


## Uninstall a Deployment

To uninstall, run:

```sh
helm uninstall rag -n rag
```

The chart does not remove NIMCache or PVCs by default. To remove them, run:

```sh
kubectl delete nimcache --all -n rag
kubectl delete pvc --all -n rag
```

## (Optional) Enable Persistence

1. Edit [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) for the persistence you want:

    - **NIM LLM** – See [NIM LLM storage](https://docs.nvidia.com/nim/large-language-models/latest/deploy-helm.html#storage). Update the `nim-llm` section in `values.yaml` as required.

    - **NeMo Retriever embedding** – See [NeMo Retriever Text Embedding storage](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/deploying.html#storage). Update the `nvidia-nim-llama-32-nv-embedqa-1b-v2` section as required.

    - **NeMo Retriever reranking** – See [NeMo Retriever Text Reranking storage](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/deploying.html#storage). Update the `nvidia-nim-llama-32-nv-rerankqa-1b-v2` section as required.

2. Run the command in [Change a Deployment](#change-a-deployment).



## Troubleshooting Helm Issues

For Helm deployment issues, see [Troubleshooting](troubleshooting.md).

:::{note}
For non-default NIM LLM profiles, see [NIM Model Profile Configuration](model-profiles.md).
:::

## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md)
- [Multi-Turn Conversation Support](multiturn.md)
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshooting](troubleshooting.md)
- [Notebooks](notebooks.md)

<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Deploy NVIDIA RAG Blueprint on Kubernetes with Helm

Use the following documentation to deploy the [NVIDIA RAG Blueprint](readme.md) on a Kubernetes cluster by using Helm.

- To deploy the Helm chart with MIG support, refer to [RAG Deployment with MIG Support](./mig-deployment.md).
- To deploy with Helm from the repository, refer to [Deploy Helm from the repository](deploy-helm-from-repo.md).
- For other deployment options, refer to [Deployment Options](readme.md#deployment-options-for-rag-blueprint).

The following are the core services that you install:

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

Plan for additional space if you are enabling persistence for multiple services.
:::

1. [Get an API Key](api-key.md).

2. Verify that you meet the [hardware requirements](support-matrix.md).

3. Verify that you have the NGC CLI available on your client computer. You can download the CLI from <https://ngc.nvidia.com/setup/installers/cli>.

4. Verify that you have Kubernetes v1.34.2 installed and running on Ubuntu 22.04/24.04. For more information, see [Kubernetes documentation](https://kubernetes.io/docs/setup/) and [NVIDIA Cloud Native Stack](https://github.com/NVIDIA/cloud-native-stack).

5. Verify that you have installed Helm 3.  To install Helm 3 (and avoid Helm 4), follow the official Helm v3 installation instructions for your platform, for example by using the `get-helm-3` script described in the [Helm documentation](https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3).

6. Verify that you have a default storage class available in the cluster for PVC provisioning. One option is the local path provisioner by Rancher.   Refer to the [installation](https://github.com/rancher/local-path-provisioner?tab=readme-ov-file#installation) section of the README in the GitHub repository.

    ```console
    kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.26/deploy/local-path-storage.yaml
    kubectl get pods -n local-path-storage
    kubectl get storageclass
    ```

7. If the local path storage class is not set as default, you can make it default by running the following code.

    ```
    kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
    ```

8. Verify that you have installed the NVIDIA GPU Operator by using the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html).

9. (Optional) You can enable time slicing for sharing GPUs between pods. For details, refer to [Time-Slicing GPUs in Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html).

10. Verify that you have installed the NVIDIA NIM Operator. If not, install it by running the following code:

    ```sh
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
      --username='$oauthtoken' \
      --password=$NGC_API_KEY
    helm repo update
    helm install nim-operator nvidia/k8s-nim-operator -n nim-operator --create-namespace
    ```

    For more details, see instructions [here](https://docs.nvidia.com/nim-operator/latest/install.html).


## Deploy the RAG Helm chart

:::{important}
When you use the Helm NIM Operator deployment, it takes approximately 60 to 70 minutes for the entire pipeline to reach a running state.
:::

To deploy End-to-End RAG Server and Ingestor Server, use the following procedure.

1. Create a namespace for the deployment by running the following code.

    ```sh
    kubectl create namespace rag
    ```

2. Install the Helm chart by running the following command.

    ```sh
    helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.5.0.tgz \
    --username '$oauthtoken' \
    --password "${NGC_API_KEY}" \
    --set imagePullSecret.password=$NGC_API_KEY \
    --set ngcApiSecret.password=$NGC_API_KEY
    ```

   :::{important}
   **For NVIDIA RTX6000 Pro Deployments:**
   
    If you are deploying on NVIDIA RTX6000 Pro GPUs (instead of H100 GPUs), you need to configure the NIM LLM model profile. The required configuration is already present but commented out in the [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) file.

    Uncomment and modify the following section under `nimOperator.nim-llm.model`:
    ```yaml
    model:
      engine: tensorrt_llm
      precision: "fp8"
      qosProfile: "throughput"
      tensorParallelism: "1"
      gpus:
        - product: "rtx6000_blackwell_sv"
    ```
   
   Then install using the modified values.yaml:
   ```sh
   helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.5.0.tgz \
     --username '$oauthtoken' \
     --password "${NGC_API_KEY}" \
     --set imagePullSecret.password=$NGC_API_KEY \
     --set ngcApiSecret.password=$NGC_API_KEY \
     -f deploy/helm/nvidia-blueprint-rag/values.yaml
   ```
   :::

   :::{note}
   Refer to [NIM Model Profile Configuration](model-profiles.md) for using non-default NIM LLM profile.
   :::


## Verify a Deployment

To verify a deployment, use the following procedure.

1. List the pods by running the following code.

    ```sh
    kubectl get pods -n rag
    ```

    You should see output similar to the following.

   :::{note}
   If some pods remain in `Pending` state after deployment, refer to [PVCs in Pending state (StorageClass issues)](troubleshooting.md#pvcs-in-pending-state-storageclass-issues) in the troubleshooting guide.
   :::

    ```sh
    NAME                                                 READY   STATUS      RESTARTS   AGE
    ingestor-server-6cc886bcdf-6rfwm                     1/1     Running     0          54m
    milvus-standalone-7dd5db4755-ctqzg                   1/1     Running     0          54m
    nemotron-embedding-ms-86f75c8f65-dfhd2          1/1     Running     0          39m
    nemoretriever-graphic-elements-v1-67d9d65bdc-ftbkw   1/1     Running     0          33m
    nemoretriever-ocr-v1-78f56cddb9-f4852                1/1     Running     0          40m
    nemoretriever-page-elements-v3-56ddcf9b4b-qsg82      1/1     Running     0          49m
    nemotron-ranking-ms-5ff774889f-fwrlm            1/1     Running     0          40m
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
   With the latest Helm NIM Operator deployment, approximately 60 to 70 minutes is required for the entire pipeline to come up into a running state. This includes time for the following:
   
   - Downloading NIM model caches (largest time component, ~40-50 minutes)
   - NIMService initialization (~10-15 minutes)
   - Pod startup and readiness checks (~5-10 minutes)

   Model downloads do not show detailed progress indicators in pod status. Pods may appear in "ContainerCreating" or "Init" state for extended periods while models download in the background.

   You can monitor the deployment progress by running the following code.

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
   
   Subsequent deployments are significantly faster (~10-15 minutes) because model caches are already populated.
   :::

2.  List services by running the following code.

    ```sh
    kubectl get svc -n rag
    ```

    You should see output similar to the following.

    ```sh
    NAME                                TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)              AGE
    ingestor-server                     ClusterIP   10.107.12.217    <none>        8082/TCP             54m
    milvus                              ClusterIP   10.99.110.203    <none>        19530/TCP,9091/TCP   54m
    nemotron-embedding-ms          ClusterIP   10.104.99.15     <none>        8000/TCP,8001/TCP    54m
    nemoretriever-graphic-elements-v1   ClusterIP   10.96.115.45     <none>        8000/TCP,8001/TCP    54m
    nemoretriever-ocr-v1                ClusterIP   10.100.107.215   <none>        8000/TCP,8001/TCP    54m
    nemoretriever-page-elements-v3      ClusterIP   10.102.237.196   <none>        8000/TCP,8001/TCP    54m
    nemotron-ranking-ms            ClusterIP   10.96.114.244    <none>        8000/TCP,8001/TCP    54m
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


## Port-Forwarding to Access Web User Interface

- [RAG UI](user-interface.md) – Run the following code to port-forward the RAG UI service to your local machine. Then access the RAG UI at `http://localhost:3000`.

  ```sh
  kubectl port-forward -n rag service/rag-frontend 3000:3000 --address 0.0.0.0
  ```

:::{note}
Port-forwarding is provided as a quick method to try out the UI. However, large file ingestion or bulk ingestion through the UI might not work due to port-forwarding timeout issues.
::: 

## Experiment with the Web User Interface

1. Open a web browser and access the RAG UI. You can start experimenting by uploading docs and asking questions. For details, see [User Interface for NVIDIA RAG Blueprint](user-interface.md).


## Change a Deployment

To change an existing deployment, after you modify the [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) file, run the following code.

```sh
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.5.0.tgz \
--username '$oauthtoken' \
--password "${NGC_API_KEY}" \
--set imagePullSecret.password=$NGC_API_KEY \
--set ngcApiSecret.password=$NGC_API_KEY \
-f deploy/helm/nvidia-blueprint-rag/values.yaml
```


## Uninstall a Deployment

To uninstall a deployment, run the following code.

```sh
helm uninstall rag -n rag
```

Run the following code to remove the NIMCache and Persistent Volume Claims (PVCs) created by the chart which are not removed by default.

```sh
kubectl delete nimcache --all -n rag
kubectl delete pvc --all -n rag
```

## (Optional) Enable Persistence

1. Update the [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) file for the persistence that you want. Use the following instructions.

    - **NIM LLM** – To enable persistence for NIM LLM, refer to [NIM LLM](https://docs.nvidia.com/nim/large-language-models/latest/deploy-helm.html#storage). Update the required fields in the `nim-llm` section of the [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) file.

    - **Nemo Retriever** – To enable persistence for Nemo Retriever embedding, refer to [Nemo Retriever Text Embedding](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/deploying.html#storage). Update the required fields in the `nvidia-nim-llama-32-nv-embedqa-1b-v2` section of the [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) file.

    - **Nemo Retriever reranking** – To enable persistence for Nemo Retriever reranking, refer to [Nemo Retriever Text Reranking](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/deploying.html#storage). Update the required fields in the `nvidia-nim-llama-32-nv-rerankqa-1b-v2` section of the [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) file.

2. Run the code in [Change a Deployment](#change-a-deployment).



## Troubleshooting Helm Issues

For troubleshooting issues with Helm deployment, refer to [Troubleshooting](troubleshooting.md).

:::{note}
Refer to [NIM Model Profile Configuration](model-profiles.md) for using non-default NIM LLM profile.
:::



## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [Multi-Turn Conversation Support](multiturn.md)
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)

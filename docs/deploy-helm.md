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

1. [Get an API Key](api-key.md).

2. Verify that you meet the [hardware requirements](support-matrix.md).

3. Verify that you have the NGC CLI available on your client computer. You can download the CLI from <https://ngc.nvidia.com/setup/installers/cli>.

4. Verify that you have Kubernetes v1.33 installed and running on Ubuntu 22.04/24.04. For more information, see [Kubernetes documentation](https://kubernetes.io/docs/setup/) and [NVIDIA Cloud Native Stack repository](https://github.com/NVIDIA/cloud-native-stack/).

5. Verify that you have installed Helm 3, see [Helm 3 Installation](https://helm.sh/docs/v3/intro/install) 

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

    ```bash
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
      --username='$oauthtoken' \
      --password=$NGC_API_KEY
    helm repo update
    helm install nim-operator nvidia/k8s-nim-operator -n nim-operator --create-namespace
    ```

    For more details, see instructions [here](https://docs.nvidia.com/nim-operator/latest/install.html).


## Deploy the RAG Helm chart

To deploy End-to-End RAG Server and Ingestor Server, use the following procedure.

1. Create a namespace for the deployment by running the following code.

    ```sh
    kubectl create namespace rag
    ```

2. Install the Helm chart by running the following command.

    ```sh
    helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.4.0-rc1.tgz \
    --username '$oauthtoken' \
    --password "${NGC_API_KEY}" \
    --set imagePullSecret.password=$NGC_API_KEY \
    --set ngcApiSecret.password=$NGC_API_KEY
    ```

   :::{note}
   Refer to [NIM Model Profile Configuration](model-profiles.md) for using non-default NIM LLM profile.
   :::

### Deploy RAG with GPU Sharing Using Dynamic Resource Allocation (DRA)

Below steps are leverages [NVIDIA DRA](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/dra-intro-install.html) with [NIM Operator](https://docs.nvidia.com/nim-operator/latest/dra.html)

>[!TIP]
>
>With DRA Setup, All NIM Service can run on 3 GPUs with atleast 80GB memory, it could be A100 or H100 or B200


- Prerequisite: [NVIDIA DRA Driver](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/dra-intro-install.html)

    - Kubernetes v1.33 or newer. Run the below commands to enable DRA FeatureGates on existing Kubernetes Cluster
         
         ```sh
         sudo sed -i 's/- kube-apiserver/- kube-apiserver\n    - --feature-gates=DynamicResourceAllocation=true\n    - --runtime-config=resource.k8s.io\/v1beta1=true\n    - --runtime-config=resource.k8s.io\/v1beta2=true/' /etc/kubernetes/manifests/kube-apiserver.yaml

         sudo sed -i 's/- kube-scheduler/- kube-scheduler\n    - --feature-gates=DynamicResourceAllocation=true/' /etc/kubernetes/manifests/kube-scheduler.yaml

         sudo sed -i 's/- kube-controller-manager/- kube-controller-manager\n    - --feature-gates=DynamicResourceAllocation=true/' /etc/kubernetes/manifests/kube-controller-manager.yaml

         sudo sed -i '$a\'$'\n''featureGates:\n  DynamicResourceAllocation: true' /var/lib/kubelet/config.yaml

         sudo systemctl daemon-reload; sudo systemctl restart kubelet
         ```

    - Enable CDI to the GPU Operator and wait for few minutes
         
         ```sh
         kubectl patch clusterpolicies.nvidia.com/cluster-policy --type='json' -p='[{"op": "replace", "path": "/spec/cdi/enabled", "value":true}]'
         kubectl patch clusterpolicies.nvidia.com/cluster-policy --type='json' -p='[{"op": "replace", "path": "/spec/cdi/default", "value":true}]'
         ```

    - Verify NVIDIA GPU Driver 565 or later.
         
         ```sh
         kubectl get pods -l app.kubernetes.io/component=nvidia-driver -n nvidia-gpu-operator -o name | xargs -I {} kubectl exec -n nvidia-gpu-operator  {} -- nvidia-smi
         ```

    - Install the NVIDIA DRA Driver
         
         ```sh
         helm upgrade --install --version="25.8.1" --create-namespace --namespace nvidia-dra-driver-gpu nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu -n nvidia-dra-driver-gpu --set gpuResourcesEnabledOverride=true     --set nvidiaDriverRoot=/run/nvidia/driver
         ```

Run the below command to install the RAG Blueprint with NVIDIA DRA

```sh
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.4.0-rc1.tgz \
--username '$oauthtoken' \
--password "${NGC_API_KEY}" \
--set imagePullSecret.password=$NGC_API_KEY \
--set ngcApiSecret.password=$NGC_API_KEY \
-f ./dra/values-dra.yaml
```

>[!NOTE]
>
>Refer to [NIM Model Profile Configuration](model-profiles.md) for using non-default NIM LLM profile.

## Verify a Deployment

To verify a deployment, use the following procedure.

1. List the pods by running the following code.

    ```sh
    kubectl get pods -n rag
    ```

    You should see output similar to the following.

    ```sh
    NAME                                                        READY   STATUS    RESTARTS      AGE
    ingestor-server-7bcff75fbb-s655f                            1/1     Running   0             23m
    nv-ingest-paddle-0                                          1/1     Running   0             23m
    rag-etcd-0                                                  1/1     Running   0             23m
    rag-frontend-5d6c6dc4bd-5xpcw                               1/1     Running   0             23m
    rag-milvus-standalone-5f5699dfb6-dzlhr                      1/1     Running   3 (23m ago)   23m
    rag-minio-f88fb7fd4-29fxk                                   1/1     Running   0             23m
    rag-nemoretriever-graphic-elements-v1-b6d465575-rl66q       1/1     Running   0             23m
    rag-nemoretriever-page-elements-v2-596679ff54-z2kkf         1/1     Running   0             23m
    rag-nemoretriever-table-structure-v1-748df88f86-z7mwb       1/1     Running   0             23m
    rag-nim-llm-0                                               1/1     Running   0             23m
    rag-nv-ingest-75cdb75c48-kbr7r                              1/1     Running   0             23m
    rag-nvidia-nim-llama-32-nv-embedqa-1b-v2-5b6dc664d8-8flpd   1/1     Running   0             23m
    rag-opentelemetry-collector-558b89885-c7c8j                 1/1     Running   0             23m
    rag-redis-master-0                                          1/1     Running   0             23m
    rag-redis-replicas-0                                        1/1     Running   0             23m
    rag-server-7758bbf9bd-rw2wh                                 1/1     Running   0             23m
    rag-text-reranking-nim-74c5f499cd-clcdg                     1/1     Running   0             23m
    rag-zipkin-5dc8d6d977-nqvvc                                 1/1     Running   0             23m
    ```

   :::{note}
   It takes approximately 5 minutes for all pods to come up. You can check Kubernetes events by running the following code.

   ```sh
   kubectl get events -n rag
   ```
   :::

2.  List services by running the following code.

    ```sh
    kubectl get svc -n rag
    ```

    You should see output similar to the following.

    ```sh
    NAME                                TYPE            EXTERNAL-IP   PORT(S)                                                   AGE
    ingestor-server                     ClusterIP      <none>        8082/TCP                                                  26m
    kubernetes                          ClusterIP      <none>        443/TCP                                                   4d20h
    nemoretriever-embedding-ms                   ClusterIP      <none>        8000/TCP                                                  26m
    nemoretriever-ranking-ms                     ClusterIP      <none>        8000/TCP                                                  26m
    nemoretriever-graphic-elements-v1   ClusterIP      <none>        8000/TCP,8001/TCP                                         26m
    nemoretriever-page-elements-v2      ClusterIP      <none>        8000/TCP,8001/TCP                                         26m
    nemoretriever-table-structure-v1    ClusterIP      <none>        8000/TCP,8001/TCP                                         26m
    nim-llm                             ClusterIP      <none>        8000/TCP                                                  26m
    nim-llm-sts                         ClusterIP      <none>        8000/TCP                                                  26m
    nv-ingest-paddle                    ClusterIP      <none>        8000/TCP,8001/TCP                                         26m
    nv-ingest-paddle-sts                ClusterIP      <none>        8000/TCP,8001/TCP                                         26m
    rag-etcd                            ClusterIP      <none>        2379/TCP,2380/TCP                                         26m
    rag-etcd-headless                   ClusterIP      <none>        2379/TCP,2380/TCP                                         26m
    rag-frontend                        NodePort       <none>        3000:31645/TCP                                            26m
    rag-milvus                          ClusterIP      <none>        19530/TCP,9091/TCP                                        26m
    rag-minio                           ClusterIP      <none>        9000/TCP                                                  26m
    rag-nv-ingest                       ClusterIP      <none>        7670/TCP                                                  26m
    rag-opentelemetry-collector         ClusterIP      <none>        6831/UDP,14250/TCP,14268/TCP,4317/TCP,4318/TCP,9411/TCP   26m
    rag-redis-headless                  ClusterIP      <none>        6379/TCP                                                  26m
    rag-redis-master                    ClusterIP      <none>        6379/TCP                                                  26m
    rag-redis-replicas                  ClusterIP      <none>        6379/TCP                                                  26m
    rag-server                          ClusterIP      <none>        8081/TCP                                                  26m
    rag-zipkin                          ClusterIP      <none>        9411/TCP                                                  26m
    ```


## Port-Forwarding to Access Web User Interface

- [RAG UI](user-interface.md) – Run the following code to port-forward the RAG UI service to your local machine. Then access the RAG UI at `http://localhost:3000`.

  ```sh
  kubectl port-forward -n rag service/rag-frontend 3000:3000 --address 0.0.0.0
  ```

## Experiment with the Web User Interface

1. Open a web browser and access the RAG UI. You can start experimenting by uploading docs and asking questions. For details, see [User Interface for NVIDIA RAG Blueprint](user-interface.md).


## Change a Deployment

To Change an existing deployment, after you modify the `values.yaml` file, run the following code.

```sh
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.4.0-rc1.tgz \
--username '$oauthtoken' \
--password "${NGC_API_KEY}" \
--set imagePullSecret.password=$NGC_API_KEY \
--set ngcApiSecret.password=$NGC_API_KEY \
-f nvidia-blueprint-rag/values.yaml
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

## (Optional) Configure Redis Backend

The ingestor server uses Redis for task status tracking. The `ENABLE_REDIS_BACKEND` environment variable controls whether Redis is used (default: `False`). **Redis backend is required when running multiple replicas of the ingestor-server.**

To enable Redis backend in the `values.yaml` file:
```yaml
ingestor-server:
  envVars:
    ENABLE_REDIS_BACKEND: "True"
```

Then apply the changes using the [Change a Deployment](#change-a-deployment) procedure.


## (Optional) Enable Persistence

1. Update the ***values.yaml*** file for the persistence that you want. Use the following instructions.

    - **NIM LLM** – To enable persistence for NIM LLM, refer to [NIM LLM](https://docs.nvidia.com/nim/large-language-models/latest/deploy-helm.html#storage). Update the required fields in the `nim-llm` section of the ***values.yaml*** file.

    - **Nemo Retriever** – To enable persistence for Nemo Retriever embedding, refer to [Nemo Retriever Text Embedding](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/deploying.html#storage). Update the required fields in the `nvidia-nim-llama-32-nv-embedqa-1b-v2` section of the ***values.yaml*** file.

    - **Nemo Retriever reranking** – To enable persistence for Nemo Retriever reranking, refer to [Nemo Retriever Text Reranking](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/deploying.html#storage). Update the required fields in the `text-reranking-nim` section of the ***values.yaml*** file.

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

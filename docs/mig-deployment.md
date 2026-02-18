<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Deploy NVIDIA RAG Blueprint on Kubernetes with Helm and MIG Support

Use this guide to deploy the [NVIDIA RAG Blueprint](readme.md) Helm chart with NVIDIA MIG (Multi-Instance GPU) slices for fine-grained GPU allocation. For other options, see [Deployment Options](readme.md#deployment-options-for-rag-blueprint).

To confirm GPU compatibility with MIG, see the [MIG Supported Hardware List](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#mig-user-guide).


## Prerequisites

Before deploying, ensure you have:

* A Kubernetes cluster with NVIDIA H100 GPUs

   :::{note}
   This section uses the `NVIDIA H100 80GB HBM3` GPU. The `mig-config.yaml` profiles are specific to this GPU. For other GPU types, see the [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/).
   :::

:::{important}
- At least 200GB free disk space per node for NIM model caches and application data
- First-time deployment takes 60–70 minutes; model downloads do not show progress indicators

To monitor progress, see [Deploy on Kubernetes with Helm](deploy-helm.md#verify-the-deployment).
:::

1. [Get an API Key](api-key.md).

2. Ensure you meet the [hardware requirements](support-matrix.md).

3. Ensure the NGC CLI is available on your client. Download it from [NGC CLI installers](https://ngc.nvidia.com/setup/installers/cli).

4. Ensure a supported Kubernetes version (for example, v1.28 or later) is installed and running on Ubuntu 22.04 or 24.04. See [Kubernetes documentation](https://kubernetes.io/docs/setup/) and [NVIDIA Cloud Native Stack 17.0](https://github.com/NVIDIA/cloud-native-stack/tree/25.12.0).

5. Install Helm 3 (not Helm 4). Follow the [Helm v3 installation](https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3) instructions for your platform (for example, the `get-helm-3` script).

6. Ensure a default storage class exists for PVC provisioning. One option is the [Rancher local path provisioner](https://github.com/rancher/local-path-provisioner?tab=readme-ov-file#installation).

    ```console
    kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.26/deploy/local-path-storage.yaml
    kubectl get pods -n local-path-storage
    kubectl get storageclass
    ```

7. If the local path storage class is not the default, set it with:

    ```sh
    kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
    ```

8. Install the [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) if you have not already.

9. (Optional) Enable [time slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html) to share GPUs between pods.

10. [Clone the RAG Blueprint Git repository](deploy-docker-self-hosted.md#clone-the-rag-blueprint-git-repository) to get the MIG configuration files.

11. Install the NVIDIA NIM Operator if needed. Run:

    ```sh
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
      --username='$oauthtoken' \
      --password=$NGC_API_KEY
    helm repo update
    helm install nim-operator nvidia/k8s-nim-operator -n nim-operator --create-namespace
    ```

    For details, see [NIM Operator installation](https://docs.nvidia.com/nim-operator/latest/install.html).



## Step 1: Enable MIG with Mixed Strategy

1. Change to the `deploy/helm/` directory:

   ```sh
   cd deploy/helm/
   ```

2. Create the deployment namespace:

   ```sh
   kubectl create namespace rag
   ```

3. Set the GPU Operator ClusterPolicy to use the mixed MIG strategy:

    ```bash
    kubectl patch clusterpolicies.nvidia.com/cluster-policy \
    --type='json' \
    -p='[{"op":"replace", "path":"/spec/mig/strategy", "value":"mixed"}]'
    ```



## Step 2: Apply the MIG Configuration

Edit [`mig-config.yaml`](../deploy/helm/mig-slicing/mig-config.yaml) to adjust the slicing pattern. The following example uses mixed MIG slice sizes on the same GPU.


:::{note}
This example uses 7× 1g.10gb on GPU 0, 2× 1g.20gb + 1× 3g.40gb on GPU 1, and 1× 7g.80gb on GPU 3. You can combine different MIG slice sizes on a single GPU for better utilization.
:::

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: custom-mig-config
data:
  config.yaml: |
    version: v1
    mig-configs:
      all-disabled:
        - devices: all
          mig-enabled: false

      custom-7x1g10-2x1g20-1x3g40-1x7g80:
        - devices: [0]
          mig-enabled: true
          mig-devices:
            "1g.10gb": 7
        - devices: [1]
          mig-enabled: true
          mig-devices:
            "1g.20gb": 2
            "3g.40gb": 1
        - devices: [3]
          mig-enabled: true
          mig-devices:
            "7g.80gb": 1
```

Apply the MIG ConfigMap and update the ClusterPolicy:

```bash
kubectl apply -n nvidia-gpu-operator -f mig-slicing/mig-config.yaml
kubectl patch clusterpolicies.nvidia.com/cluster-policy \
  --type='json' \
  -p='[{"op":"replace", "path":"/spec/migManager/config/name", "value":"custom-mig-config"}]'
```

Label the node with the MIG configuration:

```bash
kubectl label nodes <node-name> nvidia.com/mig.config=custom-7x1g10-2x1g20-1x3g40-1x7g80 --overwrite
```

Verify the MIG configuration:

```bash
kubectl get node <node-name> -o=jsonpath='{.metadata.labels}' | jq . | grep mig
```

Expected output:

```json
"nvidia.com/mig.config.state": "success"
"nvidia.com/mig-1g.10gb.count": "7"
"nvidia.com/mig-1g.20gb.count": "2"
"nvidia.com/mig-3g.40gb.count": "1"
"nvidia.com/mig-7g.80gb.count": "1"
```



## Step 3: Install RAG Blueprint Helm Chart with MIG Values

Ensure your NGC API key is set in the environment (for example, `export NGC_API_KEY="<your-key>"`). See [Get an API Key](api-key.md).

Install the RAG Blueprint Helm chart:

```bash
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.4.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f mig-slicing/values-mig.yaml
```

:::{important}
**NVIDIA RTX 6000 Pro deployments**

For NVIDIA RTX 6000 Pro GPUs (instead of H100), configure the NIM LLM model profile. The required configuration is in [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) but commented out. Uncomment and adjust the following under `nimOperator.nim-llm.model`:

```yaml
model:
  engine: tensorrt_llm
  precision: "fp8"
  qosProfile: "throughput"
  tensorParallelism: "1"
  gpus:
    - product: "rtx6000_blackwell_sv"
```

Then install with the modified values file and MIG values:
```sh
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.4.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f values.yaml \
  -f mig-slicing/values-mig.yaml
```
:::

:::{note}
For non-default NIM LLM profiles, see [NIM Model Profile Configuration](model-profiles.md).
:::

:::{note}
Because of a known MIG limitation, the ingestion profile is scaled down when using MIG slicing. Bulk ingestion, especially large jobs, may be slower or fail.
:::



## Step 4: Verify MIG Resource Allocation

To view pod GPU assignments, run [kubectl-view-allocations](https://github.com/davidB/kubectl-view-allocations):

```bash
kubectl-view-allocations
```

Expected output:

```
Resource                                    Requested   Limit    Allocatable  Free
nvidia.com/mig-1g.10gb                      (86%) 6.0   (86%) 6.0     7.0        1.0
├─ milvus-standalone-...                   1.0     1.0
├─ nemoretriever-embedding-ms-...          1.0     1.0
├─ rag-nv-ingest-...                       1.0     1.0
├─ nemoretriever-graphic-elements-v1-...   1.0     1.0
├─ nemoretriever-page-elements-v3-...      1.0     1.0
└─ nemoretriever-table-structure-v1-...    1.0     1.0

nvidia.com/mig-1g.20gb                      (100%) 2.0  (100%) 2.0     2.0        0.0
├─ nemoretriever-ranking-ms-...            1.0     1.0
└─ <other-workload>                        1.0     1.0

nvidia.com/mig-3g.40gb                      (100%) 1.0  (100%) 1.0     1.0        0.0
└─ nemoretriever-ocr-v1-...                1.0     1.0

nvidia.com/mig-7g.80gb                      (100%) 1.0  (100%) 1.0     1.0        0.0
└─ nim-llm-...                             1.0     1.0
```



## Step 5: Check the MIG Slices

From the GPU Operator driver pod, run `nvidia-smi` to inspect MIG slices:

```bash
kubectl exec -n gpu-operator -it <driver-daemonset-pod> -- nvidia-smi -L
```

Expected output:

```
GPU 0: NVIDIA H100 80GB HBM3 (UUID: ...)
  MIG 1g.10gb     Device 0: ...
  MIG 1g.10gb     Device 1: ...
  MIG 1g.10gb     Device 2: ...
  MIG 1g.10gb     Device 3: ...
  MIG 1g.10gb     Device 4: ...
  MIG 1g.10gb     Device 5: ...
  MIG 1g.10gb     Device 6: ...
GPU 1: NVIDIA H100 80GB HBM3 (UUID: ...)
  MIG 1g.20gb     Device 0: ...
  MIG 1g.20gb     Device 1: ...
  MIG 3g.40gb     Device 2: ...
GPU 3: NVIDIA H100 80GB HBM3 (UUID: ...)
  MIG 7g.80gb     Device 0: ...
```



## Step 6: Follow the Remaining Instructions

Complete the deployment using [Deploy on Kubernetes with Helm](deploy-helm.md):

- [Verify the Deployment](deploy-helm.md#verify-the-deployment)
- [Port-Forward to Access the Web UI](deploy-helm.md#port-forward-to-access-the-web-ui)
- [Experiment with the Web UI](deploy-helm.md#experiment-with-the-web-ui)
- [Change a Deployment](deploy-helm.md#change-a-deployment)
- [Uninstall a Deployment](deploy-helm.md#uninstall-a-deployment)
- [(Optional) Enable Persistence](deploy-helm.md#optional-enable-persistence)
- [Troubleshooting Helm Issues](deploy-helm.md#troubleshooting-helm-issues)



## Best Practices

- Use the `mixed` MIG strategy.
- Confirm `nvidia.com/mig.config.state` is `success` before deploying.
- Customize `values-mig.yaml` with the correct MIG GPU resource requests per pod.

## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshooting](troubleshooting.md)
- [Notebooks](notebooks.md)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)
- [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [Best Practices for Common Settings](accuracy_perf.md)

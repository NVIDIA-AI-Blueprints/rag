<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Deploy NVIDIA RAG Blueprint on Kubernetes with Helm and MIG Support

Use this documentation to deploy the [NVIDIA RAG Blueprint](readme.md) Helm chart with NVIDIA MIG (Multi-Instance GPU) slices for fine-grained GPU allocation.
For other deployment options, refer to [Deployment Options](readme.md#deployment-options-for-rag-blueprint).

To ensure that your GPUs are compatible with MIG,
refer to the [MIG Supported Hardware List](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#mig-user-guide).


## Prerequisites

Before you deploy, verify that you have the following:

* A Kubernetes cluster with NVIDIA H100 GPUs

   :::{note}
   This section showcases MIG support for `NVIDIA H100 80GB HBM3` GPU. The MIG profiles used in the `mig-config.yaml` are specific to this GPU.
   Refer to the [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) for MIG profiles of other GPU types.
   :::

:::{important}
- Ensure that you have at least 200GB of available disk space per node for NIM model caches and application data
- First-time deployment takes 60-70 minutes while large models download without visible progress indicators

For monitoring deployment progress, refer to [Deploy on Kubernetes with Helm](./deploy-helm.md#verify-a-deployment).
:::

1. [Get an API Key](api-key.md).

2. Verify that you meet the [hardware requirements](support-matrix.md).

3. Verify that you have the NGC CLI available on your client computer. You can download the CLI from <https://ngc.nvidia.com/setup/installers/cli>.

4. Verify that you have Kubernetes v1.34.2 installed and running on Ubuntu 22.04/24.04. For more information, see [Kubernetes documentation](https://kubernetes.io/docs/setup/) and [NVIDIA Cloud Native Stack 17.0](https://github.com/NVIDIA/cloud-native-stack/tree/17.0).

5. Verify that you have installed Helm 3. To install Helm 3 (and avoid Helm 4), follow the official Helm v3 installation instructions for your platform, for example by using the `get-helm-3` script described in the [Helm documentation](https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3For).

6. Verify that you have a default storage class available in the cluster for PVC provisioning. One option is the local path provisioner by Rancher.   Refer to the [installation](https://github.com/rancher/local-path-provisioner?tab=readme-ov-file#installation) section of the README in the GitHub repository.

    ```console
    kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.26/deploy/local-path-storage.yaml
    kubectl get pods -n local-path-storage
    kubectl get storageclass
    ```

6. If the local path storage class is not set as default, you can make it default by running the following code.

    ```
    kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
    ```

7. Verify that you have installed the NVIDIA GPU Operator by using the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html).

8. (Optional) You can enable time slicing for sharing GPUs between pods. For details, refer to [Time-Slicing GPUs in Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html).

9. [Clone the RAG Blueprint Git repository](deploy-docker-self-hosted.md#clone-the-rag-blueprint-git-repository) to get access to the MIG configuration files.

10. Verify that you have installed the NVIDIA NIM Operator. If not, install it by running the following code:

    ```sh
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
      --username='$oauthtoken' \
      --password=$NGC_API_KEY
    helm repo update
    helm install nim-operator nvidia/k8s-nim-operator -n nim-operator --create-namespace
    ```

    For more details, see instructions [here](https://docs.nvidia.com/nim-operator/latest/install.html).



## Step 1: Enable MIG with Mixed Strategy

1. Change your directory to ***deploy/helm/*** by running the following code.

   ```sh
   cd deploy/helm/
   ```

2. Create a namespace for the deployment by running the following code.

    ```sh
    kubectl create namespace rag
    ```

3. Update the GPU Operator's ClusterPolicy to use the mixed MIG strategy by running the following code.

    ```bash
    kubectl patch clusterpolicies.nvidia.com/cluster-policy \
    --type='json' \
    -p='[{"op":"replace", "path":"/spec/mig/strategy", "value":"mixed"}]'
    ```



## Step 2: Apply the MIG configuration

Edit the MIG configuration file [`mig-config.yaml`](../deploy/helm/mig-slicing/mig-config.yaml) to adjust the slicing pattern as needed.
The following example enables a custom configuration with mixed MIG slice sizes on the same GPU.


:::{note}
This example uses a custom slicing strategy: 7 slices of 1g.10gb on GPU 0, mixed slices (2x 1g.20gb + 1x 3g.40gb) on GPU 1, and 1 slice of 7g.80gb on GPU 3. This demonstrates the ability to combine different MIG slice sizes on a single GPU for optimal resource utilization.
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

Apply the custom MIG configuration configMap to the node and update the ClusterPolicy, by running the following code.

```bash
kubectl apply -n nvidia-gpu-operator -f mig-slicing/mig-config.yaml
kubectl patch clusterpolicies.nvidia.com/cluster-policy \
  --type='json' \
  -p='[{"op":"replace", "path":"/spec/migManager/config/name", "value":"custom-mig-config"}]'
```

Label the node with MIG configuration, by running the following code.

```bash
kubectl label nodes <node-name> nvidia.com/mig.config=custom-7x1g10-2x1g20-1x3g40-1x7g80 --overwrite
```

Verify that the MIG configuration is successfully applied, by running the following code.

```bash
kubectl get node <node-name> -o=jsonpath='{.metadata.labels}' | jq . | grep mig
```

You should see output similar to the following.

```json
"nvidia.com/mig.config.state": "success"
"nvidia.com/mig-1g.10gb.count": "7"
"nvidia.com/mig-1g.20gb.count": "2"
"nvidia.com/mig-3g.40gb.count": "1"
"nvidia.com/mig-7g.80gb.count": "1"
```



## Step 3: Install RAG Blueprint Helm Chart with MIG Values

Run the following code to install the RAG Blueprint Helm Chart.

```bash
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.4.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f mig-slicing/values-mig.yaml
```

:::{important}
**For NVIDIA RTX6000 Pro Deployments:**

If you are deploying on NVIDIA RTX6000 Pro GPUs (instead of H100 GPUs), you need to configure the NIM LLM model profile. The required configuration is already present but commented out in the [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) file.

Uncomment and modify the following section under `nimOperator.nim-llm.model` in [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml):
```yaml
model:
  engine: tensorrt_llm
  precision: "fp8"
  qosProfile: "throughput"
  tensorParallelism: "1"
  gpus:
    - product: "rtx6000_blackwell_sv"
```

Then install using the modified values.yaml along with MIG values:
```sh
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.4.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f values.yaml \
  -f mig-slicing/values-mig.yaml
```
:::

:::{note}
Refer to [NIM Model Profile Configuration](model-profiles.md) for using non-default NIM LLM profile.
:::

:::{note}
Due to a known issue with MIG support, currently the ingestion profile has been scaled down while deploying the chart with MIG slicing.
This is expected to affect the ingestion performance during bulk ingestion, specifically large bulk ingestion jobs might fail.
:::



## Step 4: Verify MIG Resource Allocation

To view pod GPU assignments, run [`kubectl-view-allocations`](https://github.com/davidB/kubectl-view-allocations) as shown following.

```bash
kubectl-view-allocations
```

You should see output similar to the following.

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

To check the MIG slices, run the following code from the GPU Operator driver pod.
This runs `nvidia-smi` within the pod to check GPU MIG slices.

```bash
kubectl exec -n gpu-operator -it <driver-daemonset-pod> -- nvidia-smi -L
```

You should see output similar to the following.

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


6. Follow the remaining instructions in [Deploy on Kubernetes with Helm](./deploy-helm.md):

    - [Verify a Deployment](deploy-helm.md#verify-a-deployment)
    - [Port-Forwarding to Access Web User Interface](deploy-helm.md#port-forwarding-to-access-web-user-interface)
    - [Experiment with the Web User Interface](deploy-helm.md#experiment-with-the-web-user-interface)
    - [Change a deployment](deploy-helm.md#change-a-deployment)
    - [Uninstall a deployment](deploy-helm.md#uninstall-a-deployment)
    - [(Optional) Enable Persistence](deploy-helm.md#optional-enable-persistence)
    - [Troubleshooting Helm Issues](deploy-helm.md#troubleshooting-helm-issues)



## Best Practices

* Ensure you have the correct MIG strategy (`mixed`) configured.
* Verify that `nvidia.com/mig.config.state` is `success` before deploying.
* Customize `values-mig.yaml` to specify the correct MIG GPU resource requests for each pod.



## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)
- [NVIDIA GPU Operator Docs](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)
- [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [Best Practices for Common Settings](accuracy_perf.md).

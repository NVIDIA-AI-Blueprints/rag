<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Deploy NVIDIA RAG Blueprint on Kubernetes with Helm and MIG Support

Use this documentation to deploy the [NVIDIA RAG Blueprint](readme.md) Helm chart with NVIDIA MIG (Multi-Instance GPU) slices for fine-grained GPU allocation.
For other deployment options, refer to [Deployment Options](readme.md#deployment-options-for-rag-blueprint).

To ensure that your GPUs are compatible with MIG,
refer to the [MIG Supported Hardware List](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#mig-supported-gpus).


## Prerequisites

Before you deploy, verify that you have the following:

* A Kubernetes cluster with NVIDIA H100 GPUs

    > [!NOTE]
    > This section showcases MIG support for `NVIDIA H100 80GB HBM3` GPU. The MIG profiles used in the `mig-config.yaml` are specific to this GPU.
    > Refer to the [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) for MIG profiles of other GPU types.

1. [Get an API Key](api-key.md).

2. Verify that you meet the [hardware requirements](support-matrix.md).

3. Verify that you have the NGC CLI available on your client computer. You can download the CLI from <https://ngc.nvidia.com/setup/installers/cli>.

4. Verify that you have Kubernetes v1.33 installed and running on Ubuntu 22.04. For more information, see [Kubernetes documentation](https://kubernetes.io/docs/setup/) and [NVIDIA Cloud Native Stack repository](https://github.com/NVIDIA/cloud-native-stack/).

5. Verify that you have a default storage class available in the cluster for PVC provisioning. One option is the local path provisioner by Rancher.   Refer to the [installation](https://github.com/rancher/local-path-provisioner?tab=readme-ov-file#installation) section of the README in the GitHub repository.

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
The following example enables a balanced configuration.


> [!NOTE]
> This example uses a balanced slicing strategy:  3 slices of 2g.20gb on GPU 0, 3 slices of 2g.20gb on GPU 1, 3 slices of 2g.20gb on GPU 2 and 1 slice of 7g.80gb on GPU 3.

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

      balanced-3-20gb-3-20gb-3-20gb-1-80gb:
        - devices: [0]
          mig-enabled: true
          mig-devices:
            "2g.20gb": 3
        - devices: [1]
          mig-enabled: true
          mig-devices:
            "2g.20gb": 3
        - devices: [2]
          mig-enabled: true
          mig-devices:
            "2g.20gb": 3
        - devices: [3]
          mig-enabled: true
          mig-devices:
            "7g.80gb": 1
```

Apply the custom MIG configuration configMap to the node and update the ClusterPolicy, by running the following code.

```bash
kubectl apply -n gpu-operator -f mig-slicing/mig-config.yaml
kubectl patch clusterpolicies.nvidia.com/cluster-policy \
  --type='json' \
  -p='[{"op":"replace", "path":"/spec/migManager/config/name", "value":"custom-mig-config"}]'
```

Label the node with MIG configuration, by running the following code.

```bash
kubectl label nodes <node-name> nvidia.com/mig.config=balanced-3-20gb-3-20gb-3-20gb-1-80gb --overwrite
```

Verify that the MIG configuration is successfully applied, by running the following code.

```bash
kubectl get node <node-name> -o=jsonpath='{.metadata.labels}' | jq . | grep mig
```

You should see output similar to the following.

```json
"nvidia.com/mig.config.state": "success"
"nvidia.com/mig-2g.20gb.count": "9"
"nvidia.com/mig-7g.80gb.count": "1"
```



## Step 3: Install RAG Blueprint Helm Chart with MIG Values

Run the following code to install the RAG Blueprint Helm Chart.

```bash
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f mig-slicing/values-mig.yaml
```

> [!NOTE]
> If the `rag-nim-llm-0` is in a `CrashLoopBackOff` after deployment, then set the model profile explicitly to avoid any errros with NIM LLM pod deployment.
> Refer to [NIM Model Profile Configuration](model-profiles.md) to set NIM LLM profile according to the GPU type and count.

> [!NOTE]
> Due to a known issue with MIG support, currently the ingestion profile has been scaled down while deploying the chart with MIG slicing.
> This is expected to affect the ingestion performance during bulk ingestion, specifically large bulk ingestion jobs might fail.



## Step 4: Verify MIG Resource Allocation

To view pod GPU assignments, run [`kubectl-view-allocations`](https://github.com/davidB/kubectl-view-allocations) as shown following.

```bash
kubectl-view-allocations
```

You should see output similar to the following.

```
Resource                                    Requested   Limit    Allocatable  Free
nvidia.com/mig-2g.20gb                      (100%) 9.0   (100%) 9.0     3.0        0.0
├─ rag-nvidia-nim-llama-...                1.0     1.0
├─ rag-text-reranking-nim-...              1.0     1.0
├─ milvus-standalone-...                   1.0     1.0
├─ nv-ingest-paddle-...                    1.0     1.0
├─ rag-nemoretriever-graphic-...           1.0     1.0
├─ rag-nemoretriever-page-...              1.0     1.0
└─ rag-nemoretriever-table-...             1.0     1.0

nvidia.com/mig-7g.80gb                      (100%) 1.0  (100%) 1.0     1.0        0.0
└─ rag-nim-llm-0                            1.0     1.0
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
  MIG 2g.20gb     Device 0: ...
  ...
GPU 1: NVIDIA H100 80GB HBM3 (UUID: ...)
  MIG 2g.20gb     Device 0: ...
  ...
GPU 2: NVIDIA H100 80GB HBM3 (UUID: ...)
  MIG 2g.20gb     Device 0: ...
  ...
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

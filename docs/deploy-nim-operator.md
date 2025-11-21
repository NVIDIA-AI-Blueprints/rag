<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Deploy NVIDIA RAG Blueprint on Kubernetes with NIM Operator

Use the following documentation to deploy the [NVIDIA RAG Blueprint](readme.md) by using NIM Operator.
For other deployment options, refer to [Deployment Options](readme.md#deployment-options-for-rag-blueprint).


## Prerequisites

1. [Get an API Key](api-key.md).

2. [Install NIM Operator 3.0](https://docs.nvidia.com/nim-operator/latest/install.html).

3. Ensure you meet [the hardware requirements](./support-matrix.md).

4. Verify that you have the NGC CLI available on your client computer. You can download the CLI from <https://ngc.nvidia.com/setup/installers/cli>.

5. Verify that you have Kubernetes v1.33 installed and running on Ubuntu 22.04. For more information, see [Kubernetes documentation](https://kubernetes.io/docs/setup/) and [NVIDIA Cloud Native Stack repository](https://github.com/NVIDIA/cloud-native-stack/).

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



## Enable NIM Operator with Helm Chart

1. Change your directory to ***deploy/helm/*** by running the following code.

   ```sh
   cd deploy/helm/
   ```

2. Create a namespace for the deployment by running the following code.

    ```sh
    kubectl create namespace rag
    ```

3. Create ImagePullSecrets with below commands

   ```sh
   kubectl create secret -n rag docker-registry ngc-secret \
   --docker-server=nvcr.io \
   --docker-username='$oauthtoken' \
   --docker-password=$NGC_API_KEY

   kubectl create secret -n rag generic ngc-api-secret \
   --from-literal=NGC_API_KEY=$NGC_API_KEY
   ```

4. Create a NIM Cache with available storage class on the cluster.

  ```sh
  kubectl apply -f deploy/helm/nim-operator/rag-nimcache.yaml -n rag
  ```

5. Now create a NIM Services

   - **No GPU Sharing**

      ```sh
      kubectl apply -f deploy/helm/nim-operator/rag-nimservice.yaml -n rag
      ```

   - **GPU Sharing with Dynamic Resource Allocation(DRA)**

      > [!TIP] With DRA Setup, All NIM Service can run on 3 GPUs with atleast 80GB memory, it could be A100 or H100 or B200

      - Prerequisite: [NVIDIA DRA Driver](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/25.3.4/dra-intro-install.html)

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
         helm upgrade --install --version="25.3.2" --create-namespace --namespace nvidia-dra-driver-gpu nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu -n nvidia-dra-driver-gpu --set gpuResourcesEnabledOverride=true     --set nvidiaDriverRoot=/run/nvidia/driver
         ```


      ```sh
      kubectl apply -f deploy/helm/nim-operator/rag-nimservice-dra.yaml -n rag
      ```

   - **List available profiles for your system for NIM LLM container**

      More details about profiles can be found [here](https://docs.nvidia.com/nim/large-language-models/latest/profiles.html).

   - **Configure NIM Model Profile**

      For optimal performance, configure the NIM model profile. See the [NIM Model Profile Configuration](model-profiles.md) section for detailed instructions and hardware-specific examples.

      Configure the `NIM_MODEL_PROFILE` in `deploy/helm/nim-operator/rag-nimservice.yaml`:

      ```yaml
      storage:
        nimCache:
          name: nemotron-llama3-49b-super
           profile: ''
        sharedMemorySizeLimit: 16Gi
      env:
      - name: NIM_MODEL_PROFILE
        value: "tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput-2321:10de-6343e21ba5cccf783d18951c6627c207b81803c3c45f1e8b59eee062ed350143-1"  # Example for H100 NVL
      ```

      After modifying the profile, reapply the NIM service:

      ```sh
      kubectl apply -f deploy/helm/nim-operator/rag-nimservice.yaml -n rag
      ```

6. Wait a few minutes and ensure that the NIMService status is `Ready` before proceeding to the next steps.

   ```sh
   kubectl get nimservice -n rag
   NAME                                STATUS     AGE
   nemoretriever-embedding-ms          Ready      20m
   nemoretriever-reranking-ms          Ready      20m
   nemoretriever-graphic-elements-v1   Ready      20m
   nemoretriever-page-elements-v2      Ready      20m
   nemoretriever-table-structure-v1    Ready      20m
   nim-llm                             Ready      20m

7. Delete the existing secret as it's conflict with RAG installation

  ```sh
  kubectl delete secret ngc-secret -n rag
  ```

8. Use the following `values-nim-operator.yaml` to deploy the RAG with NIM Operator NIM Services

  ```sh
  helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f deploy/helm/nim-operator/values-nim-operator.yaml
  ```



## Port-Forwarding to Access Web User Interface

For Helm deployments, to port-forward the the [RAG UI](user-interface.md) service to your local computer, run the following code. Then access the RAG UI at `http://localhost:3000`.

  ```sh
  kubectl port-forward -n rag service/rag-frontend 3000:3000 --address 0.0.0.0
  ```



## Experiment with the Web User Interface

1. Open a web browser and access the RAG UI. You can start experimenting by uploading docs and asking questions. For details, see [User Interface for NVIDIA RAG Blueprint](user-interface.md).



## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [Notebooks](notebooks.md)

<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Deploy NVIDIA RAG Blueprint on Kubernetes with Helm from the repository

Use the following documentation to deploy the [NVIDIA RAG Blueprint](readme.md) by using the helm chart from the repository. 

- To deploy the Helm chart with MIG support, refer to [RAG Deployment with MIG Support](./mig-deployment.md). 
- To deploy with Helm from the repository, refer to [Deploy Helm from the repository](deploy-helm-from-repo.md).
- For other deployment options, refer to [Deployment Options](readme.md#deployment-options-for-rag-blueprint).

The following are the core services that you install:

- RAG server
- Ingestor server
- NV-Ingest


## Prerequisites

Verify that you meet the prerequisites specified in [prerequisites](./deploy-helm.md#prerequisites).


## Deploy the RAG Helm chart from the repository

If you are working directly with the source Helm chart, and you want to customize components individually, use the following procedure. 

1. Change directory to `deploy/helm/` by running the following code.

   ```sh
   cd deploy/helm/
   ```

2. Create a namespace for the deployment by running the following code.

    ```sh
    kubectl create namespace rag
    ```

3. Configure Helm repo additions by editing and then running the following code.

    ```sh
    helm repo add nvidia-nim https://helm.ngc.nvidia.com/nim/nvidia/ --username='$oauthtoken' --password=$NGC_API_KEY
    helm repo add nim https://helm.ngc.nvidia.com/nim/ --username='$oauthtoken' --password=$NGC_API_KEY
    helm repo add nemo-microservices https://helm.ngc.nvidia.com/nvidia/nemo-microservices --username='$oauthtoken' --password=$NGC_API_KEY
    helm repo add baidu-nim https://helm.ngc.nvidia.com/nim/baidu --username='$oauthtoken' --password=$NGC_API_KEY
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add otel https://open-telemetry.github.io/opentelemetry-helm-charts
    helm repo add zipkin https://zipkin.io/zipkin-helm
    helm repo add prometheus https://prometheus-community.github.io/helm-charts
    ```

4. Update Helm chart dependencies by running the following code.

    ```sh
    helm dependency update nvidia-blueprint-rag
    ```

5. Install the chart by running the following code.

    ```sh
    helm upgrade --install rag -n rag nvidia-blueprint-rag/ \
    --set imagePullSecret.password=$NGC_API_KEY \
    --set ngcApiSecret.password=$NGC_API_KEY
    ```

    > [!NOTE]
    > Refer to [NIM Model Profile Configuration](model-profiles.md) to set NIM LLM profile according to the GPU type and count.
    > Set the profile explicitly to avoid any errors with NIM LLM pod deployment.


6. Follow the remaining instructions in [Deploy on Kubernetes with Helm](./deploy-helm.md):

    - [Verify a Deployment](deploy-helm.md#verify-a-deployment)
    - [Port-Forwarding to Access Web User Interface](deploy-helm.md#port-forwarding-to-access-web-user-interface)
    - [Experiment with the Web User Interface](deploy-helm.md#experiment-with-the-web-user-interface)
    - [Change a deployment](deploy-helm.md#change-a-deployment)
    - [Uninstall a deployment](deploy-helm.md#uninstall-a-deployment)
    - [(Optional) Enable Persistence](deploy-helm.md#optional-enable-persistence)
    - [Troubleshooting Helm Issues](deploy-helm.md#troubleshooting-helm-issues)



## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)

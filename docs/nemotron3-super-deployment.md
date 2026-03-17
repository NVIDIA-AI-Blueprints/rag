# Using Nemotron-3-Super-120B-A12B LLM NIM

[Nemotron-3-Super-120B-A12B](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard) is a large language model (LLM) trained by NVIDIA, designed to deliver strong agentic, reasoning, and conversational capabilities. It is optimized for collaborative agents and high-volume workloads such as IT ticket automation. This LLM can considerably improve the accuracy of the RAG pipeline, especially with reasoning enabled. ([Model card](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard))

We recommend to use the model with low-effort reasoning mode with a reasoning budget of 256 to have a balance between accuracy and performance. You can switch to non-reasoning mode for maximum performance or use reasoning mode for best accuracy.

## Hardware requirements

For Docker and Kubernetes deployment, see the following:

- **Docker (local NIM):** [Hardware Requirements (Docker)](support-matrix.md#hardware-requirements-docker)
- **Kubernetes (Helm):** [Hardware Requirements (Kubernetes)](support-matrix.md#hardware-requirements-kubernetes)

For [self-hosted local NIM](deploy-docker-self-hosted.md) deployment with `nemotron-3-super-120b-a12b`, you need one of the following:

- 3 x H100
- 3 x B200
- 3 x RTX PRO 6000

### Hardware Requirements (Kubernetes)

To deploy with [Helm](deploy-helm.md) using `nemotron-3-super-120b-a12b`, you need one of the following:

- 9 x H100-80GB
- 9 x B200
- 9 x RTX PRO 6000

---

## Start services using NVIDIA-hosted models

No local GPU needed for the LLM. The file `deploy/compose/nemotron3-super-cloud.env` sets all NVIDIA-hosted (cloud) endpoints and the `nemotron-3-super-120b-a12b` model.

1. [Set your API key](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/api-key.md) and prompt config, then source the env files:

```bash
export NGC_API_KEY=<ngc-api-key>
source deploy/compose/.env
source deploy/compose/nemotron3-super-cloud.env
export PROMPT_CONFIG_FILE=$(pwd)/deploy/compose/nemotron3-super-prompt.yaml
```

2. Follow [Start services using NVIDIA-hosted models](deploy-docker-nvidia-hosted.md#start-services-using-nvidia-hosted-models) to start the vectorstore, rag-server, and ingestor-server.

---

## Start services using self-hosted on-premises models

1. Update `nims.yaml`

   Edit `deploy/compose/nims.yaml` and change the `nim-llm` service image and GPU allocation:

   ```yaml
   nim-llm:
     image: nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b:1.8.0
     ...
     user: "0"
     environment:
       NGC_API_KEY: ${NGC_API_KEY}
       NIM_MAX_MODEL_LEN: "32768"  # required for TP2 profile
       NIM_KVCACHE_PERCENT: "0.9"
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               device_ids: ['1','2']  # 2 GPUs for FP8 TP2
               capabilities: [gpu]
   ```

   > Note: To deploy TP2 profiles you need to limit NIM_MAX_MODEL_LEN to 32768

   To confirm that a TP2 profile is available for your hardware, run:

   ```bash
   docker run -ti --rm --gpus all nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b:1.8.0 list-model-profiles
   ```

   Check the [model page](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard) for more details.

   > Note: For RTX 6000 Pro GPUs, additional NIM environment variables are required — see [RTX 6000 Pro](#rtx-6000-pro) below.

2. Set nemotron-3-super specific environment variables.

   Ensure the section **`Endpoints for using cloud NIMs`** in `deploy/compose/.env` is **commented** (so on-prem endpoints are used).

   ```bash
   source deploy/compose/.env
   source deploy/compose/nemotron3-super.env
   export PROMPT_CONFIG_FILE=$(pwd)/deploy/compose/nemotron3-super-prompt.yaml
   export LLM_MAX_TOKENS=16256
   ```

   Follow [Start services using self-hosted on-premises models](deploy-docker-self-hosted.md#start-services-using-self-hosted-on-premises-models) to start the vectorstore, rag-server, NIMs, and ingestor-server.

**RTX 6000 Pro**

> Note: To deploy TP2 profiles on RTX PRO 6000 Blackwell Server Edition, run the following commands. You don't need to go through these steps if you are using TP4 or TP8 profile.

1. Edit `/etc/default/grub` and set:

   ```text
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt"
   ```

2. Run:

   ```bash
   sudo update-grub2
   sudo reboot
   ```

3. In `nims.yaml`, add under the `nim-llm` `environment:` block:

   ```yaml
   environment:
     # In addition to variable already set in step 1
     NCCL_P2P_DISABLE: "1"
   ```

---

## Helm deployment (`nemotron-3-super-120b-a12b`)

From the repository root, run:

```bash
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.5.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml \
  -f deploy/helm/nvidia-blueprint-rag/nemotron3-super-values.yaml
```

The prompt file `deploy/compose/nemotron3-super-prompt.yaml` is tuned for `nemotron-3-super-120b-a12b`. To customize it, see [Prompt customization in Helm chart](prompt-customization.md#prompt-customization-in-helm-chart).

**RTX 6000 Pro**

> Note: To deploy TP2 profiles on RTX PRO 6000 Blackwell Server Edition, run the following commands. You don't need to go through these steps if you are using TP4 or TP8 profile.

1. Edit `/etc/default/grub` and set:

   ```text
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt"
   ```

2. Run:

   ```bash
   sudo update-grub2
   sudo reboot
   ```

3. From the repository root, run:

   ```bash
   helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.5.0.tgz \
     --username '$oauthtoken' \
     --password "${NGC_API_KEY}" \
     --set imagePullSecret.password=$NGC_API_KEY \
     --set ngcApiSecret.password=$NGC_API_KEY \
     -f deploy/helm/nvidia-blueprint-rag/values.yaml \
     -f deploy/helm/nvidia-blueprint-rag/nemotron3-super-values.yaml \
     -f deploy/helm/nvidia-blueprint-rag/nemotron3-super-rtx6000-values.yaml
   ```

---

## Reasoning and non-reasoning mode

To disable reasoning mode set following

```bash
export LLM_ENABLE_THINKING=false
export LLM_REASONING_BUDGET=0
```

For other options (e.g. full reasoning budget), see [Enable reasoning for Nemotron 3 models](enable-nemotron-thinking.md).

# Using Nemotron-3-Super LLM NIM

[Nemotron-3-Super-120B-A12B](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard) is a large language model (LLM) trained by NVIDIA, designed to deliver strong agentic, reasoning, and conversational capabilities. It is optimized for collaborative agents and high-volume workloads such as IT ticket automation. This LLM can considerably improve the accuracy of the RAG pipeline, especially with reasoning enabled. ([Model card](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard))

We recommend to use the model with low-effort reasoning mode with a reasoning budget of 256 to have a balance between accuracy and performance. You can switch to non-reasoning mode for maximum performance or use reasoning mode for best accuracy.

## Hardware requirements

For Docker and Kubernetes deployment, see the following:

- **Docker (local NIM):** [Hardware Requirements (Docker)](support-matrix.md#hardware-requirements-docker)
- **Kubernetes (Helm):** [Hardware Requirements (Kubernetes)](support-matrix.md#hardware-requirements-kubernetes)

Nemotron 3 Super requires **at least 2 GPUs** for the LLM NIM (e.g. FP8 TP2)—**3 GPUs total** for the Docker flow and **9 GPUs total** for the Helm flow. Ensure your host or cluster meets these counts before proceeding.

---

## Start services using NVIDIA-hosted models

No local GPU needed for the LLM. The file `deploy/compose/nemotron3-super-cloud.env` sets all NVIDIA-hosted (cloud) endpoints and the Nemotron 3 Super model.

1. Set your API key and prompt config, then source the env files:

```bash
export NGC_API_KEY=<ngc-api-key>
export APP_LLM_APIKEY=<llm-api-key>   # from https://build.nvidia.com/
source deploy/compose/.env
source deploy/compose/nemotron3-super-cloud.env
export PROMPT_CONFIG_FILE=$(pwd)/deploy/compose/nemotron3-super-prompt.yaml
```

2. Follow [Start services using NVIDIA-hosted models](deploy-docker-nvidia-hosted.md#start-services-using-nvidia-hosted-models) to start the vectorstore, rag-server, and ingestor-server.

Generate an API key at [build.nvidia.com](https://build.nvidia.com/). For model details, see the [Nemotron 3 Super 120B model page](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b). Default is low-effort reasoning; to use non-reasoning mode, see [Reasoning and non-reasoning mode](#reasoning-and-non-reasoning-mode).

---

## Start services using self-hosted on-premises models

Local NIM deployment for Nemotron 3 Super requires **at least 2 GPUs** for the LLM NIM (e.g. FP8 TP2). **Total GPU requirement for the Docker flow: 3 GPUs.** Ensure your host has the required GPUs before proceeding.

### 1. Update `nims.yaml`

Edit `deploy/compose/nims.yaml` and change the `nim-llm` service image and GPU allocation:

```yaml
nim-llm:
  image: nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b:1.8.0
  ...
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['1','2']  # 2 GPUs for FP8 TP2
            capabilities: [gpu]
```

You may need to set `USERID=0`. To confirm that a TP2 profile is available for your hardware, run:

```bash
docker run -ti --rm --gpus all nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b:1.8.0 list-model-profiles
```

### 2. Set NIM environment

No changes to the NIM environment are required for the default flow. The default `nims.yaml` already sets `NGC_API_KEY` for the `nim-llm` service in the same format:

```yaml
environment:
  NGC_API_KEY: ${NGC_API_KEY}
```

Only if you use **RTX 6000 Pro** or encounter **OOM** do you need to add `NIM_MAX_MODEL_LEN` and related variables—see [RTX 6000 Pro (Docker / local)](#rtx-6000-pro-docker--local) and [Notes](#notes).

### 3. Source env and deploy

Ensure the section **`Endpoints for using cloud NIMs`** in `deploy/compose/.env` is **commented** (so on-prem endpoints are used). See [Start services using self-hosted on-premises models](deploy-docker-self-hosted.md#start-services-using-self-hosted-on-premises-models) for the same step.

```bash
source deploy/compose/.env
source deploy/compose/nemotron3-super.env
export PROMPT_CONFIG_FILE=$(pwd)/deploy/compose/nemotron3-super-prompt.yaml
# then start compose as usual
```

Follow the instructions in [Start services using self-hosted on-premises models](deploy-docker-self-hosted.md#start-services-using-self-hosted-on-premises-models) to start the vectorstore, rag-server, NIMs, and ingestor-server.

Default is low-effort reasoning. To use non-reasoning mode, see [Reasoning and non-reasoning mode](#reasoning-and-non-reasoning-mode).

- **Model**: `nvidia/nemotron-3-super-120b-a12b` — see the [model page](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b) for documentation.

---

## Notes

- **NIM_MAX_MODEL_LEN** is needed mainly for **RTX 6000 Pro** or when **OOM** occurs. In those cases, add it to the NIM environment (see [RTX 6000 Pro (Docker / local)](#rtx-6000-pro-docker--local)) and set the RAG server **LLM_MAX_TOKENS** as below. The default flow does not require it.
- **LLM_MAX_TOKENS**: When using a reduced NIM max context (e.g. `NIM_MAX_MODEL_LEN: 32768` for RTX 6000 Pro or OOM), set the RAG server **LLM_MAX_TOKENS** lower value such as **16256** to avoid issues:
  For Docker, you can set this in `deploy/compose/nemotron3-super.env`: the file includes a commented `LLM_MAX_TOKENS` line—uncomment and set the value when using reduced max model length. For Helm, set `envVars.LLM_MAX_TOKENS` in `values.yaml`.
- **BF16** requires minimum TP4 (4 GPUs). **FP8** can run on TP2 (2 GPUs).
- **NIM_MODEL_PROFILE** must be specified; use `list-model-profiles` inside the container to find available profiles for your hardware. If that does not work, try setting `NIM_KVCACHE_PERCENT: 0.9`.
- **Reasoning**: Default is low-effort reasoning. For non-reasoning mode, see [Reasoning and non-reasoning mode](#reasoning-and-non-reasoning-mode). For other options (e.g. full reasoning budget), see [Enable reasoning for Nemotron 3 models](enable-nemotron-thinking.md).

---

## RTX 6000 Pro (Docker / local)

On **NVIDIA RTX 6000 Pro** GPUs, use the following.

**Host (one-time)**  
Edit `/etc/default/grub` and set:

```text
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt"
```

Then run `sudo update-grub2` and `sudo reboot`.

**NIM environment**  
In `nims.yaml`, add or set under the `nim-llm` service `environment:` block (same format as the default):

```yaml
environment:
  NGC_API_KEY: ${NGC_API_KEY}
  NCCL_P2P_DISABLE: "1"
  NIM_MAX_MODEL_LEN: "32768"
  NIM_KVCACHE_PERCENT: "0.9"
```

**RAG server**  
When using this reduced max model length, set **LLM_MAX_TOKENS** appropriately: **16256**. For Docker, uncomment and set `LLM_MAX_TOKENS` in `deploy/compose/nemotron3-super.env`.

---

## Helm deployment (Nemotron 3 Super)

For Helm-based deployment of Nemotron 3 Super, follow the general flow in [Change the LLM or Embedding Model – For Helm Deployments](change-model.md#for-helm-deployments). The file [`deploy/helm/nvidia-blueprint-rag/values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) contains commented blocks for Nemotron 3 Super—uncomment the relevant sections for your scenario as described below. After editing `values.yaml`, apply changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

**GPU requirement**: Nemotron 3 Super requires **at least 2 GPUs** for the LLM NIM. **Total GPU requirement for the Helm flow: 9 GPUs.**

### Model and resource settings (all hardware)

In `values.yaml`, for **all hardware** (e.g. RTX 6000 Pro, B200):

1. **NIM LLM image**: Uncomment the "For Nemotron 3 Super" image block under `nimOperator.nim-llm.image` (repository `nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b`, tag `1.8.0`) and comment the default image block above it.

2. **NIM_SERVED_MODEL_NAME**: Uncomment the "For Nemotron 3 Super" `NIM_SERVED_MODEL_NAME` line under `nimOperator.nim-llm.env` (value `nvidia/nemotron-3-super-120b-a12b`) and comment the default line above it. In the rag-server `envVars` section, set `APP_LLM_MODELNAME` to `nvidia/nemotron-3-super-120b-a12b`.

3. **Model block**: Under `nimOperator.nim-llm.model`, comment `engine: tensorrt_llm` and uncomment the "For Nemotron 3 Super" block (`engine: vllm`, `precision: "fp8"`, `tensorParallelism: "2"`).

4. **Resources**: Uncomment the "For Nemotron 3 Super" resources block under `nimOperator.nim-llm.resources` (2 GPUs) and comment the default 1-GPU resources block above it.

### RTX 6000 Pro (Helm)

In addition to the [model and resource settings](#model-and-resource-settings-all-hardware) above:

- **Host**: Same as Docker: set `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt"` in `/etc/default/grub`, then `sudo update-grub2` and `sudo reboot`.
- **values.yaml**: Under `nimOperator.nim-llm.env`, uncomment the "For Nemotron 3 Super on RTX 6000 Pro or OOM" block (`NIM_MAX_MODEL_LEN: "32768"`, `NCCL_P2P_DISABLE`, `NIM_KVCACHE_PERCENT`). If the default `NIM_MAX_MODEL_LEN` is present, comment it and use the values from the uncommented block.

- **RAG server (LLM_MAX_TOKENS)**: When using this reduced max model length on RTX 6000 Pro, uncomment the `LLM_MAX_TOKENS` line in the rag-server `envVars` section (the one with the "For Nemotron 3 Super with reduced context" comment) and set it to **16256** for reasoning or **1024** for non-reasoning; comment the default `LLM_MAX_TOKENS` line above it.

### Deploy with updated values

After you have made all the changes above, deploy or upgrade using your updated `values.yaml`. Follow [Change a Deployment](deploy-helm.md#change-a-deployment) in the Helm deployment guide for the exact commands. For a new installation, see [Deploy on Kubernetes with Helm](deploy-helm.md).

---

## Reasoning and non-reasoning mode

The Nemotron 3 Super env files (`nemotron3-super.env` and `nemotron3-super-cloud.env`) enable **low-effort reasoning** by default for better accuracy.

To use **non-reasoning mode** (maximum speed or minimal latency), set **`LLM_ENABLE_THINKING=false`** before starting services. Setting the reasoning budget to `0` alone does not disable reasoning; the pipeline does not pass budget `0` to the LLM and the model uses its default. To disable reasoning, always set `LLM_ENABLE_THINKING=false`.

```bash
export LLM_ENABLE_THINKING=false
export LLM_REASONING_BUDGET=0
```

For other options (e.g. full reasoning budget), see [Enable reasoning for Nemotron 3 models](enable-nemotron-thinking.md).

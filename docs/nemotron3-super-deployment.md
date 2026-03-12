# Using Nemotron-3-Super LLM NIM

[Nemotron-3-Super-120B-A12B](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard) is a large language model (LLM) trained by NVIDIA, designed to deliver strong agentic, reasoning, and conversational capabilities. It is optimized for collaborative agents and high-volume workloads such as IT ticket automation. This LLM can considerably improve the accuracy of the RAG pipeline, especially with reasoning enabled. ([Model card](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard))

We recommend to use the model with low-effort reasoning mode with a reasoning budget of 256 to have a balance between accuracy and performance. You can switch to non-reasoning mode for maximum performance or use reasoning mode for best accuracy.

## Hardware requirements

For Docker and Kubernetes deployment, see the following:

- **Docker (local NIM):** [Hardware Requirements (Docker)](support-matrix.md#hardware-requirements-docker)
- **Kubernetes (Helm):** [Hardware Requirements (Kubernetes)](support-matrix.md#hardware-requirements-kubernetes)

For [self-hosted local NIM](deploy-docker-self-hosted.md) deployment with Nemotron 3 Super, you need one of the following (2 GPUs for the LLM NIM, FP8 TP2; 3 GPUs total including pipeline):

- 3 x H100
- 3 x B200
- 3 x RTX PRO 6000

### Hardware Requirements (Kubernetes)

To deploy with [Helm](deploy-helm.md) using Nemotron 3 Super, you need one of the following (9 GPUs total):

- 9 x H100-80GB
- 9 x B200
- 9 x RTX PRO 6000

---

## Start services using NVIDIA-hosted models

No local GPU needed for the LLM. The file `deploy/compose/nemotron3-super-cloud.env` sets all NVIDIA-hosted (cloud) endpoints and the Nemotron 3 Super model.

1. Set your API key and prompt config, then source the env files:

```bash
export NGC_API_KEY=<ngc-api-key>
source deploy/compose/.env
source deploy/compose/nemotron3-super-cloud.env
export PROMPT_CONFIG_FILE=$(pwd)/deploy/compose/nemotron3-super-prompt.yaml
```

2. Follow [Start services using NVIDIA-hosted models](deploy-docker-nvidia-hosted.md#start-services-using-nvidia-hosted-models) to start the vectorstore, rag-server, and ingestor-server.

Generate an API key at [build.nvidia.com](https://build.nvidia.com/). For model details, see the [Nemotron 3 Super 120B model page](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b). Default is low-effort reasoning; to use non-reasoning mode, see [Reasoning and non-reasoning mode](#reasoning-and-non-reasoning-mode).

---

## Start services using self-hosted on-premises models

Local NIM deployment for Nemotron 3 Super requires **2 GPUs** for the LLM NIM (FP8 TP2), for 2 × H100, 2 × B200, or 2 × RTX PRO 6000. **Total GPU requirement for the Docker flow: 3 GPUs.** Ensure your host has the required GPUs before proceeding.

### 1. Update `nims.yaml`

Edit `deploy/compose/nims.yaml` and change the `nim-llm` service image and GPU allocation:

```yaml
nim-llm:
  image: nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b:1.8.0
  ...
  user: "0"
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            device_ids: ['1','2']  # 2 GPUs for FP8 TP2
            capabilities: [gpu]
```

To confirm that a TP2 profile is available for your hardware, run:

```bash
docker run -ti --rm --gpus all nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b:1.8.0 list-model-profiles
```

### 2. Set NIM environment

No changes to the NIM environment are required for the default flow. The default `nims.yaml` already sets `NGC_API_KEY` for the `nim-llm` service in the same format:

```yaml
environment:
  NGC_API_KEY: ${NGC_API_KEY}
```

Only if you use **RTX 6000 Pro** you need to add `NIM_MAX_MODEL_LEN` and related variables—see [RTX 6000 Pro (Docker / local)](#rtx-6000-pro-docker--local) and [Notes](#notes).

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

- **NIM_MAX_MODEL_LEN** is required for **RTX 6000 Pro**. Add it to the NIM environment (see [RTX 6000 Pro (Docker / local)](#rtx-6000-pro-docker--local)) and set the RAG server **LLM_MAX_TOKENS** as below. The default flow does not require it.
- **LLM_MAX_TOKENS**: When using reduced NIM max context on RTX 6000 Pro (e.g. `NIM_MAX_MODEL_LEN: 32768`), set the RAG server **LLM_MAX_TOKENS** to a lower value such as **16256** to avoid issues. For Docker, export before starting: `export LLM_MAX_TOKENS=16256` (or `1024` for non-reasoning). You can also uncomment and set the value in `deploy/compose/nemotron3-super.env`. For Helm, set `envVars.LLM_MAX_TOKENS` in `values.yaml`.
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
When using this reduced max model length, set **LLM_MAX_TOKENS** appropriately. For Docker, export before starting the rag-server:

```bash
export LLM_MAX_TOKENS=16256   # for reasoning; use 1024 for non-reasoning
```

Alternatively, uncomment and set `LLM_MAX_TOKENS` in `deploy/compose/nemotron3-super.env`.

---

## Helm deployment (Nemotron 3 Super)

From the repository root, run one of the following commands.


```bash
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.5.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml \
  -f deploy/helm/nvidia-blueprint-rag/nemotron3-super-values.yaml
```

**RTX 6000 Pro only** (after [host GRUB/reboot steps](#rtx-6000-pro-helm)):

```bash
helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvstaging/blueprint/charts/nvidia-blueprint-rag-v2.5.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  -f deploy/helm/nvidia-blueprint-rag/values.yaml \
  -f deploy/helm/nvidia-blueprint-rag/nemotron3-super-values.yaml \
  -f deploy/helm/nvidia-blueprint-rag/nemotron3-super-rtx6000-values.yaml
```

Override files: [`nemotron3-super-values.yaml`](../deploy/helm/nvidia-blueprint-rag/nemotron3-super-values.yaml) (model, image, resources, env); [`nemotron3-super-rtx6000-values.yaml`](../deploy/helm/nvidia-blueprint-rag/nemotron3-super-rtx6000-values.yaml) (RTX 6000 Pro NIM env and `LLM_MAX_TOKENS`). For a chart from the repo instead of NGC, use the same `-f` options with your `helm upgrade --install` command—see [Change a Deployment](deploy-helm.md#change-a-deployment).

---

## Reasoning and non-reasoning mode

The Nemotron 3 Super env files (`nemotron3-super.env` and `nemotron3-super-cloud.env`) enable **low-effort reasoning** by default for better accuracy.

To use **non-reasoning mode** (maximum speed or minimal latency), set **`LLM_ENABLE_THINKING=false`** before starting services. Setting the reasoning budget to `0` alone does not disable reasoning; the pipeline does not pass budget `0` to the LLM and the model uses its default. To disable reasoning, always set `LLM_ENABLE_THINKING=false`.

```bash
export LLM_ENABLE_THINKING=false
export LLM_REASONING_BUDGET=0
```

For other options (e.g. full reasoning budget), see [Enable reasoning for Nemotron 3 models](enable-nemotron-thinking.md).

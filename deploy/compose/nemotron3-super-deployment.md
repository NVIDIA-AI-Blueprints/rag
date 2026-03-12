# Nemotron 3 Super - Deployment Guide

Nemotron 3 Super is a larger model with different deployment requirements. For **local NIM deployment**, the LLM requires **at least 2 GPUs** (FP8 TP2)—one more than the default single-GPU LLM. **Total GPU requirement: 3 GPUs** for the Docker flow (local NIM) and **9 GPUs** for the Helm flow. The guide uses **low-effort reasoning as the default** for better accuracy; you can switch to non-reasoning for maximum speed.

Required files are in the repository: `deploy/compose/.env`, `deploy/compose/nemotron3-super.env` (local), `deploy/compose/nemotron3-super-cloud.env` (cloud), and `deploy/compose/nemotron3-super-prompt.yaml`.

---

## Option A: NVIDIA Inference Hub (Cloud)

No local GPU needed for the LLM. Uses NVIDIA-hosted endpoint. Use the same approach as the main deployment guides: source `deploy/compose/.env` first, then the Nemotron 3 Super overrides for cloud.

1. Open `deploy/compose/.env` and **uncomment** the section **`Endpoints for using cloud NIMs`** (the block that sets cloud API URLs and model names). Leave the on-prem endpoints commented when using cloud. See [Deploy with Docker (NVIDIA-Hosted Models)](../../docs/deploy-docker-nvidia-hosted.md) for the same step.

2. Source the environment and set the Nemotron 3 Super model and API key:

```bash
export NGC_API_KEY=<ngc-api-key>
source deploy/compose/.env
source deploy/compose/nemotron3-super-cloud.env
export PROMPT_CONFIG_FILE=$(pwd)/deploy/compose/nemotron3-super-prompt.yaml
```

Generate an API key at [build.nvidia.com](https://build.nvidia.com/). For model details and documentation, see the [Nemotron 3 Super 120B model page](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b).

Use the path above from the repo root, or set `PROMPT_CONFIG_FILE` to the absolute path to `nemotron3-super-prompt.yaml` in your clone.

Default is low-effort reasoning. To use non-reasoning mode, see [Reasoning and non-reasoning mode](#reasoning-and-non-reasoning-mode).

Then follow the instructions to start the vectorstore, rag-server, and ingestor-server from [Deploy with Docker (NVIDIA-Hosted Models)](../../docs/deploy-docker-nvidia-hosted.md).

---

## Option B: Local NIM Deployment

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

Ensure the section **`Endpoints for using cloud NIMs`** in `deploy/compose/.env` is **commented** (so on-prem endpoints are used). See [Deploy with Docker (Self-Hosted Models)](../../docs/deploy-docker-self-hosted.md) for the same step.

```bash
source deploy/compose/.env
source deploy/compose/nemotron3-super.env
export PROMPT_CONFIG_FILE=$(pwd)/deploy/compose/nemotron3-super-prompt.yaml
# then start compose as usual
```

Follow the instructions in [Deploy with Docker (Self-Hosted Models)](../../docs/deploy-docker-self-hosted.md) to start the vectorstore, rag-server, NIMs, and ingestor-server.

Default is low-effort reasoning. To use non-reasoning mode, see [Reasoning and non-reasoning mode](#reasoning-and-non-reasoning-mode).

- **Model**: `nvidia/nemotron-3-super-120b-a12b` — see the [model page](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b) for documentation.

---

## Notes

- **NIM_MAX_MODEL_LEN** is needed mainly for **RTX 6000 Pro** or when **OOM** occurs. In those cases, add it to the NIM environment (see [RTX 6000 Pro (Docker / local)](#rtx-6000-pro-docker--local)) and set the RAG server **LLM_MAX_TOKENS** as below. The default flow does not require it.
- **LLM_MAX_TOKENS**: When using a reduced NIM max context (e.g. `NIM_MAX_MODEL_LEN: 32768` for RTX 6000 Pro or OOM), set the RAG server **LLM_MAX_TOKENS** lower value such as **16256** to avoid issues:
  For Docker, you can set this in `deploy/compose/nemotron3-super.env`: the file includes a commented `LLM_MAX_TOKENS` line—uncomment and set the value when using reduced max model length. For Helm, set `envVars.LLM_MAX_TOKENS` in `values.yaml`.
- **BF16** requires minimum TP4 (4 GPUs). **FP8** can run on TP2 (2 GPUs).
- **NIM_MODEL_PROFILE** must be specified; use `list-model-profiles` inside the container to find available profiles for your hardware. If that does not work, try setting `NIM_KVCACHE_PERCENT: 0.9`.
- **Reasoning**: Default is low-effort reasoning. For non-reasoning mode, see [Reasoning and non-reasoning mode](#reasoning-and-non-reasoning-mode). For other options (e.g. full reasoning budget), see [Enable reasoning for Nemotron 3 models](../../docs/enable-nemotron-thinking.md).

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

For Helm-based deployment of Nemotron 3 Super, follow the general flow in [Change the LLM or Embedding Model – For Helm Deployments](../../docs/change-model.md#for-helm-deployments), then apply the model-specific steps below. After editing `values.yaml`, apply changes as described in [Change a Deployment](../../docs/deploy-helm.md#change-a-deployment).

**NIM LLM image**: Set the Nemotron 3 Super image in `values.yaml` under `nimOperator.nim-llm.image`:

- **repository**: `nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b`
- **tag**: `1.8.0`

Set `NIM_SERVED_MODEL_NAME` (and `APP_LLM_MODELNAME` in the rag-server section) to `nvidia/nemotron-3-super-120b-a12b`.

**GPU requirement**: Nemotron 3 Super requires **at least 2 GPUs** for the LLM NIM. **Total GPU requirement for the Helm flow: 9 GPUs.**

### Model and resource settings (all hardware)

Apply the following in `values.yaml` for **all hardware** (e.g. RTX 6000 Pro, B200):

1. **Model block**: Comment out the block that sets:
   - `model.engine: vllm`
   - `precision: "fp8"`
   - `tensorParallelism: "2"`

2. **Resources**: Set the LLM NIM to use 2 GPUs under `nimOperator.nim-llm.resources`:

   ```yaml
   resources:
     limits:
       nvidia.com/gpu: 2
     requests:
       nvidia.com/gpu: 2
   ```

### RTX 6000 Pro (Helm)

In addition to the [model and resource settings](#model-and-resource-settings-all-hardware) above:

- **Host**: Same as Docker: set `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt"` in `/etc/default/grub`, then `sudo update-grub2` and `sudo reboot`.
- **values.yaml**: Add these env vars under `nimOperator.nim-llm.env`:

  ```yaml
  - name: NIM_MAX_MODEL_LEN
    value: "32768"
  - name: NCCL_P2P_DISABLE
    value: "1"
  - name: NIM_KVCACHE_PERCENT
    value: "0.9"
  ```

- **RAG server (LLM_MAX_TOKENS)**: When using this reduced max model length on RTX 6000 Pro, you must set **LLM_MAX_TOKENS** in `values.yaml` to avoid errors. In the rag-server section, set `envVars.LLM_MAX_TOKENS` to **16256**.

### Deploy with updated values

After you have made all the changes above, deploy or upgrade using your updated `values.yaml`. Follow [Change a Deployment](../../docs/deploy-helm.md#change-a-deployment) in the Helm deployment guide for the exact commands. For a new installation, see [Deploy on Kubernetes with Helm](../../docs/deploy-helm.md).

---

## Reasoning and non-reasoning mode

The Nemotron 3 Super env files (`nemotron3-super.env` and `nemotron3-super-cloud.env`) enable **low-effort reasoning** by default for better accuracy.

To use **non-reasoning mode** (maximum speed or minimal latency), set the following before starting services:

```bash
export LLM_ENABLE_THINKING=false
export LLM_REASONING_BUDGET=0
```

For other options (e.g. full reasoning budget), see [Enable reasoning for Nemotron 3 models](../../docs/enable-nemotron-thinking.md).

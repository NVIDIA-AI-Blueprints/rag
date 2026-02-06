<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Model Profiles for NVIDIA RAG Blueprint

Use the following documentation to learn about model profiles available for [NVIDIA RAG Blueprint](readme.md).

This section provides the recommended model profiles for different hardware configurations. 
You should use these profiles for all deployment methods (Docker Compose, Helm Chart, RAG python library, and NIM Operator).


## Profile Selection Guidelines

- **TensorRT-LLM profiles** (`tensorrt_llm-*`) are recommended for best performance
- For multi-GPU setups, ensure proper GPU allocation by setting `LLM_MS_GPU_ID` environment variable in docker setup.
- Always verify available profiles using the `list-model-profiles` command before deployment
- By default, NIM uses automatic profile detection. However, you can manually specify a profile for optimal performance using the instructions below



## List Available Profiles

To see all available profiles for your specific hardware configuration, run the following code.

```bash
USERID=$(id -u) docker run --rm --gpus all \
  -v ~/.cache/model-cache:/opt/nim/.cache \
  nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:1.14.0 \
  list-model-profiles
```

## How to Find the Correct Profile for Your Hardware

1. **Run** the `list-model-profiles` command (see above) to see all available profiles
2. **Select** a profile from the "Compatible with system and runnable" section
3. **Choose** based on these profile name components:
   - `tensorrt_llm` = best performance (recommended), `vllm` = alternative
   - GPU type: `h100_nvl`, `h100`, `a100`, `b200`, `rtx6000_blackwell_sv`, etc.
   - Precision: `fp8` (faster) or `bf16` (better accuracy)
   - `tp<N>` = number of GPUs (e.g., `tp1` = 1 GPU, `tp2` = 2 GPUs)
   - `throughput` = batch processing, `latency` = interactive

**Example**: For 1xH100 NVL, select a profile like `tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput-...` and copy the full string from the output.

## Configuring Model Profiles

**Note:** NIM automatically detects and selects the optimal profile for your hardware. Only configure a specific profile if you experience issues with the default deployment, such as performance problems or out-of-memory errors.

### Docker Compose Deployment

To set a specific model profile in Docker Compose, add the `NIM_MODEL_PROFILE` environment variable to the `nim-llm` service in `deploy/compose/nims.yaml`:

```yaml
  nim-llm:
    container_name: nim-llm-ms
    image: nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:1.14.0
    # ... other configuration ...
    environment:
      NGC_API_KEY: ${NGC_API_KEY}
      NIM_MODEL_PROFILE: ${NIM_MODEL_PROFILE-""}  # Add this line
```

Then set the profile in your environment or `.env` file before deploying:

```bash
export NIM_MODEL_PROFILE="tensorrt_llm-h100-fp8-tp1-pp1-throughput-2330:10de-a5381c1be0b8ee66ad41e7dc7b4e6d2cffaa7a4e37ca05f57898817560b0bd2b-1"
docker compose -f deploy/compose/nims.yaml up -d
```

### Helm Deployment

For Helm deployments with NIM operator, configure the model profile declaratively through the `model` section in [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml):

```yaml
nimOperator:
  nim-llm:
    enabled: true
    replicas: 1
    service:
      name: "nim-llm"
    image:
      repository: nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5
      pullPolicy: IfNotPresent
      tag: "1.14.0"
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 1
    model:
      engine: tensorrt_llm
      precision: "fp8"
      qosProfile: "throughput"
      tensorParallelism: "1"
      gpus:
        - product: "rtx6000_blackwell_sv"  # Change based on your GPU
    storage:
      pvc:
        create: true
        size: "120Gi"
        volumeAccessMode: ReadWriteOnce
        storageClass: ""
      sharedMemorySizeLimit: "16Gi"
    env:
      - name: NIM_HTTP_API_PORT
        value: "8000"
      - name: NIM_TRITON_LOG_VERBOSE
        value: "1"
      - name: NIM_SERVED_MODEL_NAME
        value: "nvidia/llama-3.3-nemotron-super-49b-v1.5"
```

**Key profile parameters:**
- **`engine`**: `tensorrt_llm` (recommended) or `vllm`
- **`precision`**: `fp8` (faster) or `bf16` (better accuracy)
- **`qosProfile`**: `throughput` (batch processing) or `latency` (interactive)
- **`tensorParallelism`**: Number of GPUs (e.g., `"1"`, `"2"`)
- **`gpus.product`**: GPU type (e.g., `h100`, `h100_nvl`, `a100`, `rtx6000_blackwell_sv`)

:::{note}
The NIM operator automatically selects the optimal profile based on these parameters.
:::

After modifying the [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) file, apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

For detailed HELM deployment instructions, see [Helm Deployment Guide](deploy-helm.md).



## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)
- [Deploy with Helm and MIG Support](mig-deployment.md)

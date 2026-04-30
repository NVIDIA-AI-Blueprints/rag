<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Model Profiles for NVIDIA RAG Blueprint

Use the following documentation to learn about model profiles available for [NVIDIA RAG Blueprint](readme.md).

This section provides model profile guidance for different hardware configurations.
You should use these profiles for all deployment methods (Docker Compose, Helm Chart, RAG python library, and NIM Operator).


## Profile Selection Guidelines

- NIM LLM 2.0 uses vLLM as the inference backend. Use `vllm-*` profiles for Nemotron 3 Super NIM 2.0.3.
- For multi-GPU setups, ensure proper GPU allocation by setting `LLM_MS_GPU_ID`, `LLM_MS_GPU_ID2`, and related Docker Compose GPU variables.
- Always verify available profiles using the `list-model-profiles` command before deployment
- By default, NIM uses automatic profile detection. However, you can manually specify a profile for optimal performance using the instructions below



## List Available Profiles

To see all available profiles for your specific hardware configuration, run the following code.

```bash
USERID=$(id -u) docker run --rm --gpus all \
  -v ~/.cache/model-cache:/opt/nim/.cache \
  nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b:2.0.3 \
  list-model-profiles
```

## How to Find the Correct Profile for Your Hardware

1. **Run** the `list-model-profiles` command (see above) to see all available profiles
2. **Select** a profile from the "Compatible with system and runnable" section
3. **Choose** based on these profile name components:
   - Engine: `vllm`
   - Precision: `fp8`, `bf16`, or `nvfp4`, depending on the GPU
   - `tp<N>` = number of GPUs (e.g., `tp1` = 1 GPU, `tp2` = 2 GPUs)

**Example**: The default Nemotron 3 Super deployment uses an FP8 TP2 profile such as `vllm-fp8-tp2-pp1`.

## Configuring Model Profiles

**Note:** NIM automatically detects and selects the optimal profile for your hardware. Only configure a specific profile if you experience issues with the default deployment, such as performance problems or out-of-memory errors.

### Docker Compose Deployment

To set a specific model profile in Docker Compose, add the `NIM_MODEL_PROFILE` environment variable to the `nim-llm` service in `deploy/compose/nims.yaml`:

```yaml
  nim-llm:
    container_name: nim-llm-ms
    image: nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b:2.0.3
    # ... other configuration ...
    environment:
      NGC_API_KEY: ${NGC_API_KEY}
      NIM_MODEL_PROFILE: ${NIM_MODEL_PROFILE-""}  # Add this line
```

Then set the profile in your environment or `.env` file before deploying:

```bash
export NIM_MODEL_PROFILE="vllm-fp8-tp2-pp1"
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
      repository: nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b
      pullPolicy: IfNotPresent
      tag: "2.0.3"
    resources:
      limits:
        nvidia.com/gpu: 2
      requests:
        nvidia.com/gpu: 2
    model:
      engine: vllm
      precision: "fp8"
      tensorParallelism: "2"
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
      - name: NIM_LOG_LEVEL
        value: "INFO"
      - name: NIM_SERVED_MODEL_NAME
        value: "nvidia/nemotron-3-super-120b-a12b"
      - name: NIM_PASSTHROUGH_ARGS
        value: "--enable-chunked-prefill --kv-cache-dtype fp8"
```

**Key profile parameters:**
- **`engine`**: `vllm`
- **`precision`**: `fp8`, `bf16`, or `nvfp4`, based on the support matrix for your GPU
- **`tensorParallelism`**: Number of GPUs used by the LLM profile (e.g., `"1"`, `"2"`, `"4"`)
:::{note}
The NIM operator automatically selects the optimal profile based on these parameters.
:::

For the full Nemotron 3 Super NIM 2.0.3 GPU matrix and minimum GPU counts, see the [NVIDIA NIM LLM 2.0.3 support matrix](https://docs.nvidia.com/nim/large-language-models/2.0.3/reference/support-matrix.html).

After modifying the [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) file, apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

For detailed HELM deployment instructions, see [Helm Deployment Guide](deploy-helm.md).



## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)
- [Deploy with Helm and MIG Support](mig-deployment.md)

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



## List Available Profiles

To see all available profiles for your specific hardware configuration, run the following code.

```bash
USERID=$(id -u) docker run --rm --gpus all \
  -v ~/.cache/model-cache:/opt/nim/.cache \
  nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:1.13.1 \
  list-model-profiles
```

## Hardware-Specific Profiles

The following profiles are optimized for different common GPU configurations:

### 1xH100 NVL
```bash
NIM_MODEL_PROFILE=tensorrt_llm-h100_nvl-fp8-tp1-pp1-throughput-2321:10de-d347471b749e4e6b6e5956bb0f600b6646461c214cadadf6614baf305054a743-1
```

### 1xH100 SXM
```bash
NIM_MODEL_PROFILE=tensorrt_llm-h100-fp8-tp1-pp1-throughput-2330:10de-a5381c1be0b8ee66ad41e7dc7b4e6d2cffaa7a4e37ca05f57898817560b0bd2b-1
```

### 2xA100 SXM
```bash
NIM_MODEL_PROFILE=vllm-bf16-tp2-pp1-32c3b968468aefcfb3ea1db5a16e3dc9d64395f02ef68a06175e8bbdb0038601
```

### 1xRTX PRO 6000
```bash
NIM_MODEL_PROFILE=tensorrt_llm-rtx6000_blackwell_sv-fp8-tp1-pp1-throughput-2bb5:10de-d21d6986d29d8abf555f35c9a4c8146c4b10595d9e57e6efabd4a026efcc0c4a-1
```

### 2xB200
```bash
NIM_MODEL_PROFILE=tensorrt_llm-b200-fp8-tp2-pp1-throughput-2901:10de-d2ff2bbf26fdabe28afaf754ca8e5615ed337e19d873da15627c209849f51072-2
```



## Related Topics

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [Deploy with Docker (Self-Hosted Models)](deploy-docker-self-hosted.md)
- [Deploy with Docker (NVIDIA-Hosted Models)](deploy-docker-nvidia-hosted.md)
- [Deploy with Helm](deploy-helm.md)
- [Deploy with Helm and MIG Support](mig-deployment.md)
- [Deploy with NIM Operator](deploy-nim-operator.md)

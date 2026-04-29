# Nemotron-3-Super-120B-A12B

[Nemotron-3-Super-120B-A12B](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard) is the default LLM for the NVIDIA RAG Blueprint. It is trained by NVIDIA and designed to deliver strong agentic, reasoning, and conversational capabilities. It is optimized for collaborative agents and high-volume workloads such as IT ticket automation.

We recommend using the model with low-effort reasoning mode with a reasoning budget of 256 to balance accuracy and performance. You can switch to non-reasoning mode for maximum performance or use reasoning mode for best accuracy.

## Hardware requirements

For Docker and Kubernetes deployment, see the following:

- **Docker (local NIM):** [Hardware Requirements (Docker)](support-matrix.md#hardware-requirements-docker)
- **Kubernetes (Helm):** [Hardware Requirements (Kubernetes)](support-matrix.md#hardware-requirements-kubernetes)

For [self-hosted local NIM](deploy-docker-self-hosted.md) deployment with `nemotron-3-super-120b-a12b`, you need one of the following:

- 3 x H100
- 3 x B200
- 3 x RTX PRO 6000

The default Docker Compose configuration uses 2 GPUs for the LLM and 1 shared GPU for the non-LLM NIMs.

For [Helm](deploy-helm.md) deployment, you need one of the following:

- 8 x H100-80GB
- 8 x B200
- 8 x RTX PRO 6000

The default Helm configuration requests 2 GPUs for the LLM and 1 GPU for each enabled non-LLM NIM pod.
Add 1 GPU if you enable an optional GPU-backed service such as VLM generation or GPU-accelerated Milvus.
For the full platform matrix, see the [NVIDIA NIM LLM 2.0.3 support matrix](https://docs.nvidia.com/nim/large-language-models/2.0.3/reference/support-matrix.html).

---

## RTX PRO 6000 Setup

> Note: These steps are only required for RTX PRO 6000 Blackwell Server Edition using the TP2 profile. Skip if you are using a TP4 or TP8 profile.

1. Edit `/etc/default/grub` and set:

   ```text
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt"
   ```

2. Run:

   ```bash
   sudo update-grub2
   sudo reboot
   ```

No additional configuration changes are needed in `nims.yaml` or `values.yaml` beyond the defaults.

---

## Reasoning and non-reasoning mode

To disable reasoning mode:

```bash
export LLM_ENABLE_THINKING=false
export LLM_REASONING_BUDGET=0
```

For other options (e.g. full reasoning budget), see [Enable reasoning for Nemotron 3 models](enable-nemotron-thinking.md).

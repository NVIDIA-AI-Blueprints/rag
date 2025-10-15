# Quickstart for NVIDIA AI Workbench 

This blueprint is for developers who want a quick start to set up a RAG solution with a path-to-production with NVIDIA NIM.

> **Note**
> This blueprint runs in [NVIDIA AI Workbench](https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/introduction.html). It's a free, lightweight developer platform that you can run on your own systems to get up and running with complex AI applications and workloads in a short amount of time. 

> You may want to [**fork**](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository) this repository into your own account before proceeding. Otherwise you won't be able to fully push any changes you make because this NVIDIA-owned repository is **read-only**.

*Navigating the README*: [Project Overview](#project-overview) | [Get Started](#get-started) | [License](#license) | [Terms of Use](#terms-of-use)

*Other Resources*: [:arrow_down: Download AI Workbench](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/) | [:book: User Guide](https://docs.nvidia.com/ai-workbench/) |[:open_file_folder: Other Projects](https://docs.nvidia.com/ai-workbench/user-guide/latest/quickstart/example-projects.html) | [:rotating_light: User Forum](https://forums.developer.nvidia.com/t/support-workbench-example-blueprint-rag/336539)

## Project Overview

This blueprint serves as a reference solution for a foundational Retrieval Augmented Generation (RAG) pipeline. One of the key use cases in Generative AI is enabling users to ask questions and receive answers based on their enterprise data corpus. This blueprint demonstrates how to set up a RAG solution that uses NVIDIA NIM and GPU-accelerated components. By default, this blueprint leverages locally-deployed NVIDIA NIM microservices to meet specific data governance and latency requirements. However, you can replace these models with your NVIDIA-hosted models available in the [NVIDIA API Catalog](build.nvidia.com).

[Read More](../../README.md#software-components)

## Get Started

Ensure you have satisfied the prerequisites for this Blueprint ([details](../../README.md#hardware-requirements)). 

### Local Hosting 

1. Open NVIDIA AI Workbench. Select a **Location** to work in.

1. **Clone** the project with URL: https://github.com/NVIDIA-AI-Blueprints/rag

1. On the **Project Dashboard**, resolve the yellow unconfigured secrets warning by inputting your ``NGC_API_KEY``.

1. Select ``ingest``, ``rag``, ``vectordb``, and ``local`` compose profiles from the dropdown under the **Compose** section.

   * Note: ``observability`` and ``guardrails`` are optional profiles you may enable.

1. Select **Start**. The compose services may take several minutes to pull and build.

1. When all compose services are ready, access the frontend on the IP address, eg. ``http://<ip_addr>:8090``. 

1. You can now interact with the RAG Chatbot through its browser interface.

### Use Build API Endpoints

1. Open NVIDIA AI Workbench. Select a **Location** to work in.

1. **Clone** the project with URL: https://github.com/NVIDIA-AI-Blueprints/rag

1. On the **Project Dashboard**, resolve the yellow unconfigured secrets warning by inputting your ``NGC_API_KEY``.

1. Under **File Browser**, locate the ``variables.env`` file.

1. From the hamburger menu, select **Edit**. Make the following edits: 

   * Comment out the variables for ``on-prem NIMs``
   * Uncomment the variables for using ``cloud NIMs``
   * Save your changes

1. On the **Project Dashboard**, select ``ingest``, ``rag``, and ``vectordb`` compose profiles under the **Compose** section.

   * Note: ``observability`` and ``guardrails`` are optional profiles you may enable. Do not select ``local``.

1. Select **Start**. The compose services may take several minutes to pull and build.

1. When the compose services are ready, access the frontend on the IP address, eg. ``http://<ip_addr>:8090``. 

1. You can now interact with the RAG Chatbot through its browser interface.

## License

This NVIDIA AI BLUEPRINT is licensed under the [Apache License, Version 2.0.](../../LICENSE) This project will download and install additional third-party open source software projects and containers. Review [the license terms of these open source projects](../../LICENSE-3rd-party.txt) before use.

Use of the models in this blueprint is governed by the [NVIDIA AI Foundation Models Community License](https://docs.nvidia.com/ai-foundation-models-community-license.pdf).


## Terms of Use
This blueprint is governed by the [NVIDIA Agreements | Enterprise Software | NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and the [NVIDIA Agreements | Enterprise Software | Product Specific Terms for AI Product](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/). The models are governed by the [NVIDIA Agreements | Enterprise Software | NVIDIA Community Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/) and the [NVIDIA RAG dataset](https://github.com/NVIDIA-AI-Blueprints/rag/tree/v2.0.0/data/multimodal) which is governed by the [NVIDIA Asset License Agreement](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/data/LICENSE.DATA).

The following models that are built with Llama are governed by the [Llama 3.2 Community License Agreement](https://www.llama.com/llama3_2/license/): nvidia/llama-3.3-nemotron-super-49b-v1, nvidia/llama-3.2-nv-embedqa-1b-v2, and nvidia/llama-3.2-nv-rerankqa-1b-v2.

<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Notebooks for NVIDIA RAG Blueprint

This section contains Jupyter notebooks that demonstrate how to use the [NVIDIA RAG Blueprint](readme.md) APIs and advanced development features.

## Set Up the Notebook Environment

To run a notebook in a Python virtual environment, use the following procedure.

1. Create and activate a virtual environment.

    ```bash
    python3 -m virtualenv venv
    source venv/bin/activate
    ```

2. Ensure that you have JupyterLab and required dependencies installed.

    ```bash
    pip3 install jupyterlab
    ```

3. Run the following command to start JupyterLab and allow access from any IP.

    ```bash
    jupyter lab --allow-root --ip=0.0.0.0 --NotebookApp.token='' --port=8889 --no-browser
    ```

### Set-up Notes

- Ensure that API keys and credentials are correctly set up before you run a notebook.
- Modify endpoints or request parameters as necessary to match your specific use case.
- For the custom VDB operator notebook, ensure that Docker is available for running OpenSearch services.

## Run a Notebook

After your notebook environment is set up, follow these steps to run a notebook.

1. Access JupyterLab by opening a browser and navigating to `http://<your-server-ip>:8889`.
2. Navigate to the notebook and run the cells sequentially.

## Beginner Notebooks

Start with the following notebooks to learn basic API interactions.

- [ingestion_api_usage.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/notebooks/ingestion_api_usage.ipynb) – Demonstrates how to interact with the NVIDIA RAG ingestion service, including how to upload and process documents for retrieval-augmented generation (RAG).

- [retriever_api_usage.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/notebooks/retriever_api_usage.ipynb) – Demonstrates how to use the NVIDIA RAG retriever service, including different query techniques and retrieval strategies.

- [image_input.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/blob/develop/notebooks/image_input.ipynb) – Demonstrates multimodal query support, enabling you to query documents using both text and images. Covers VLM embeddings, image extraction, and the search/generate APIs with visual queries.


## Experiment with the Ingestion API Usage Notebook

After the RAG Blueprint is [deployed](../docs/readme.md#deployment-options-for-rag-blueprint), you can use the Ingestion API Usage notebook to start experimenting with it.

1. Download and install Git LFS by following the [installation instructions](https://git-lfs.com/).

2. Initialize Git LFS in your environment.

   ```bash
   git lfs install
   ```

3. Pull the dataset into the current repo.

   ```bash
   git lfs pull
   ```

4. Install jupyterlab.

   ```bash
   pip install jupyterlab
   ```

5. Use this command to run Jupyter Lab so that you can execute this IPython notebook.

   ```bash
   jupyter lab --allow-root --ip=0.0.0.0 --NotebookApp.token='' --port=8889
   ```

6. Run the [ingestion_api_usage](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/notebooks/ingestion_api_usage.ipynb) notebook. Follow the cells in the notebook to ingest the PDF files from the data/dataset folder into the vector store.



## Intermediate Notebooks

Use the following notebooks to learn comprehensive Python client usage, metadata, summarization, and other features.

- [summarization.ipynb](../notebooks/summarization.ipynb) – Demonstrates document summarization customization including page filtering, fast shallow extraction, and multiple summarization strategies. Covers both Library Mode and Docker Mode API usage.

- [evaluation_01_ragas.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/notebooks/evaluation_01_ragas.ipynb) – Evaluate your RAG system using three key metrics with the [Ragas](https://docs.ragas.io/en/stable/) library. 

- [evaluation_02_recall.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/notebooks/evaluation_02_recall.ipynb) – Evaluate retrieval performance using the recall metric, which measures the fraction of relevant documents successfully retrieved at various top-k thresholds.

- [nb_metadata.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/notebooks/nb_metadata.ipynb) – Demonstrates metadata features including metadata ingestion, filtering, and extraction. Includes step-by-step examples of how to use metadata for enhanced document retrieval and Q&A capabilities. This notebook is for users who want to implement sophisticated metadata-based filtering in their RAG applications.

- [rag_library_usage.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/notebooks/rag_library_usage.ipynb) – Demonstrates native usage of the NVIDIA RAG Python client, including environment setup, document ingestion, collection management, and querying. This notebook provides end-to-end API usage examples for interacting directly with the RAG system from Python, covering both ingestion and retrieval workflows.

- [rag_library_lite_usage.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/notebooks/rag_library_lite_usage.ipynb) – Demonstrates containerless deployment of the NVIDIA RAG Python package in lite mode. Uses Milvus Lite (embedded vector database) and NV-Ingest subprocess mode for a simplified setup without Docker containers. Leverages NVIDIA cloud APIs for embeddings, ranking, and LLM inference. **Note**: This mode does not support image/table/chart citations or document summarization.



## Advanced Notebooks

Use the following notebooks to learn how to how to extend the system with custom vector database implementations.

- [building_rag_vdb_operator.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/notebooks/building_rag_vdb_operator.ipynb) – Demonstrates how to create and integrate custom vector database (VDB) operators with the NVIDIA RAG blueprint. This notebook builds a complete OpenSearch VDB operator from scratch by using the VDBRag base class architecture. This notebook is for developers who want to extend NVIDIA RAG with their own vector database implementations.

- [mcp_server_usage.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/notebooks/mcp_server_usage.ipynb) – Demonstrates how to use the NVIDIA RAG MCP server via MCP transports (SSE, streamable-http, and stdio) instead of REST APIs. Covers launching the server, connecting with the MCP Python client, listing tools, and calling Ingestor tools (`create_collection`, `list_collections`, `upload_documents`, `get_documents`, `update_documents`, `delete_documents`, `update_collection_metadata`, `update_document_metadata`, `delete_collections`) and RAG tools (`generate`, `search`, `get_summary`).

- [image_input.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/notebooks/image_input.ipynb) – Demonstrates how to use the NVIDIA RAG retriever APIs with multimodal queries (text + images). Covers deploying VLM and multimodal embedding NIMs, ingesting documents with image extraction, and using the search and generate APIs with visual inputs for use cases like querying product catalogs with images.


## Deployment Notebooks

Use the following notebook for cloud deployment scenarios.

- [launchable.ipynb](https://github.com/NVIDIA-AI-Blueprints/rag/tree/main/notebooks/launchable.ipynb) – A deployment-ready notebook intended to run in a [Brev environment](https://console.brev.dev/environment/new). To learn more about Brev, refer to [Brev](https://docs.nvidia.com/brev/latest/about-brev.html). Follow the instructions for running Jupyter notebooks in a cloud-based environment based on the hardware requirements specified in the launchable.



## Set Up the Notebook Environment

To run a notebook, use the following procedure with [uv](https://docs.astral.sh/uv/) - a fast Python package manager.

> **Note**: Python version **3.11 or higher** is required.

1. Install uv (if not already installed):

    ```bash
    curl -LsSf https://astral.sh/uv/0.8.12/install.sh | sh
    ```

2. Create and activate a virtual environment:

    ```bash
    uv venv --python=python3.12
    source .venv/bin/activate
    ```

3. Install JupyterLab:

    ```bash
    uv pip install jupyterlab
    ```

4. Start JupyterLab:

    ```bash
    jupyter lab --allow-root --ip=0.0.0.0 --NotebookApp.token='' --port=8889 --no-browser
    ```

### Set-up Notes
- Ensure that API keys and credentials are correctly set up before you run a notebook.
- Modify endpoints or request parameters as necessary to match your specific use case.
- For the custom VDB operator notebook, ensure that Docker is available for running OpenSearch services.



## Run a Notebook

After you set up your notebook environment, to run a notebook, use the following procedure.

1. Access JupyterLab by opening a browser and navigating to `http://<your-server-ip>:8889`.
2. Navigate to the notebook and run the cells sequentially.



## Related Topics

- [Get Started](deploy-docker-self-hosted.md)
- [User Interface](user-interface.md)

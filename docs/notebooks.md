<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Notebooks for NVIDIA RAG Blueprint

This section contains Jupyter notebooks that demonstrate how to use the [NVIDIA RAG Blueprint](readme.md) APIs and advanced development features.


## Beginner Notebooks

Start with the following notebooks to learn basic API interactions.

- [ingestion_api_usage.ipynb](../notebooks/ingestion_api_usage.ipynb) – Demonstrates how to interact with the NVIDIA RAG ingestion service, including how to upload and process documents for retrieval-augmented generation (RAG).

- [retriever_api_usage.ipynb](../notebooks/retriever_api_usage.ipynb) – Demonstrates how to use the NVIDIA RAG retriever service, including different query techniques and retrieval strategies.



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

6. Run the [ingestion_api_usage](../notebooks/ingestion_api_usage.ipynb) notebook. Follow the cells in the notebook to ingest the PDF files from the data/dataset folder into the vector store.



## Intermediate Notebooks

Use the following notebooks to learn comprehensive Python client usage, metadata, and other features.

- [evaluation_01_ragas.ipynb](../notebooks/evaluation_01_ragas.ipynb) – Evaluate your RAG system using three key metrics with the [Ragas](https://docs.ragas.io/en/stable/) library. 

- [evaluation_02_recall.ipynb](../notebooks/evaluation_02_recall.ipynb) – Evaluate retrieval performance using the recall metric, which measures the fraction of relevant documents successfully retrieved at various top-k thresholds.

- [nb_metadata.ipynb](../notebooks/nb_metadata.ipynb) – Demonstrates metadata features including metadata ingestion, filtering, and extraction. Includes step-by-step examples of how to use metadata for enhanced document retrieval and Q&A capabilities. This notebook is for users who want to implement sophisticated metadata-based filtering in their RAG applications.

- [rag_library_usage.ipynb](../notebooks/rag_library_usage.ipynb) – Demonstrates native usage of the NVIDIA RAG Python client, including environment setup, document ingestion, collection management, and querying. This notebook provides end-to-end API usage examples for interacting directly with the RAG system from Python, covering both ingestion and retrieval workflows.



## Advanced Notebooks

Use the following notebooks to learn how to how to extend the system with custom vector database implementations.

- [building_rag_vdb_operator.ipynb](../notebooks/building_rag_vdb_operator.ipynb) – Demonstrates how to create and integrate custom vector database (VDB) operators with the NVIDIA RAG blueprint. This notebook builds a complete OpenSearch VDB operator from scratch by using the VDBRag base class architecture. This notebook is for developers who want to extend NVIDIA RAG with their own vector database implementations.



## Deployment Notebooks

Use the following notebook for cloud deployment scenarios.

- [launchable.ipynb](../notebooks/launchable.ipynb) – A deployment-ready notebook intended to run in a [Brev environment](https://console.brev.dev/environment/new). To learn more about Brev, refer to [Brev](https://docs.nvidia.com/brev/latest/about-brev.html). Follow the instructions for running Jupyter notebooks in a cloud-based environment based on the hardware requirements specified in the launchable.



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

After you set up your notebook environment, to run a notebook, use the following procedure.

1. Access JupyterLab by opening a browser and navigating to `http://<your-server-ip>:8889`.
2. Navigate to the notebook and run the cells sequentially.



## Related Topics

- [Get Started](deploy-docker-self-hosted.md)
- [User Interface](user-interface.md)

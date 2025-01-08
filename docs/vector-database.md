<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Vector Database Customization
<!-- TOC -->

- [Vector Database Customization](#vector-database-customization)
  - [Available Vector Databases](#available-vector-databases)
  - [Configuring Milvus with GPU Acceleration](#configuring-milvus-with-gpu-acceleration)
  - [Configuring Support for an External Milvus database](#configuring-support-for-an-external-milvus-database)
  - [Adding a New Vector Store](#adding-a-new-vector-store)

<!-- /TOC -->

## Available Vector Databases

By default, the Docker Compose files for the examples deploy Milvus as the vector database with CPU-only support.
You must install the NVIDIA Container Toolkit to use Milvus with GPU acceleration.

## Configuring Milvus with GPU Acceleration

1. Edit the `deploy/vectordb.yaml` file and make the following changes to the Milvus service.

   - Change the image tag to include the `-gpu` suffix:

     ```yaml
     milvus:
       container_name: milvus-standalone
       image: milvusdb/milvus:v2.4.4-gpu
       ...
     ```

   - Add the GPU resource reservation:

     ```yaml
     ...
     depends_on:
       - "etcd"
       - "minio"
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               capabilities: ["gpu"]
               device_ids: ['${VECTORSTORE_GPU_DEVICE_ID:-0}']
     profiles: ["nemo-retriever", "milvus", ""]
     ```

## Configuring Support for an External Milvus database

1. Edit the `docker-compose.yaml` file for the RAG example and make the following edits.

   - Remove or comment the `include` path to the `vectordb.yaml` file:

     ```yaml
     include:
       - path:
         # - ./vectordb.yaml
         - ./nims.yaml
     ```

   - To use an external Milvus server, specify the connection information and restart the containers:

     ```yaml
     environment:
       APP_VECTORSTORE_URL: "http://<milvus-hostname-or-ipaddress>:19530"
       APP_VECTORSTORE_NAME: "milvus"
       ...
     ```

## Adding a New Vector Store

You can extend the code to add support for any vector store.

1. Navigate to the file `src/utils.py` in the project's root directory.

2. Modify the `create_vectorstore_langchain` function to handle your new vector store. Implement the logic for creating your vector store object within it.

   ```python
   def create_vectorstore_langchain(document_embedder, collection_name: str = "") -> VectorStore:
      # existing code
      elif config.vector_store.name == "chromadb":
         from langchain_chroma import Chroma
         import chromadb

         logger.info(f"Using Chroma collection: {collection_name}")
         persistent_client = chromadb.PersistentClient()
         vectorstore = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=document_embedder,
         )
   ```

3. Update the `get_docs_vectorstore_langchain` function to retrieve a list of documents from your new vector store. Implement your retrieval logic within it.

   ```python
   def get_docs_vectorstore_langchain(vectorstore: VectorStore) -> List[str]:
      # Existing code
      elif  settings.vector_store.name == "chromadb":
         chroma_data = vectorstore.get()
         filenames = set([extract_filename(metadata) for metadata in chroma_data.get("metadatas", [])])
         return filenames
   ```

4. Update the `del_docs_vectorstore_langchain` function to handle document deletion in your new vector store.

   ```python
   def del_docs_vectorstore_langchain(vectorstore: VectorStore, filenames: List[str]) -> bool:
      # Existing code
      elif  settings.vector_store.name == "chromadb":
         chroma_data = vectorstore.get()
         for filename in filenames:
               ids_list = [chroma_data.get("ids")[idx] for idx, metadata in enumerate(chroma_data.get("metadatas", [])) if extract_filename(metadata) == filename]
               vectorstore.delete(ids_list)
         return True
   ```

5. In your `chains.py` implementation, import the preceding functions from `utils.py`.
   The sample `chains.py` already imports the functions.

   ```python
   from .utils import (
      create_vectorstore_langchain,
      get_docs_vectorstore_langchain,
      del_docs_vectorstore_langchain,
      get_vectorstore
   )
   ```

6. Update `requirements.txt` with any additional package required for the vector store.

   ```text
   # existing dependency
   langchain-chroma
   ```

7. Build and start the containers.

   1. Navigate to the root directory.

   2. Set the `APP_VECTORSTORE_NAME` environment variable for the `rag-server` microservice.
      Set it to the name of your newly added vector store.

      ```yaml
      export APP_VECTORSTORE_NAME: "chromadb"
      ```

   3. Build and deploy the microservices.

      ```console
      docker compose -f deploy/compose/docker-compose.yaml up -d --build
      ```

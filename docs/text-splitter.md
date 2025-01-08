<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Text Splitter Customizations
<!-- TOC -->

- [Text Splitter Customizations](#text-splitter-customizations)
  - [Adjusting Chunk Size and Overlap](#adjusting-chunk-size-and-overlap)
  - [Using a Custom Text Splitter](#using-a-custom-text-splitter)
  - [Build and Start the Container](#build-and-start-the-container)

<!-- /TOC -->

The default text splitter is a `RecursiveCharacterTextSplitter` instance.

## Adjusting Chunk Size and Overlap

The text splitter divides documents into smaller chunks for processing.
You can control the chunk size and overlap using environment variables in `rag-server` service of your `docker-compose.yaml` file:

- `APP_TEXTSPLITTER_CHUNKSIZE`: Sets the maximum number of tokens allowed in each chunk.
- `APP_TEXTSPLITTER_CHUNKOVERLAP`: Defines the number of tokens that overlap between consecutive chunks.

```yaml
services:
  rag-server:
    environment:
      APP_TEXTSPLITTER_CHUNKSIZE: 256
      APP_TEXTSPLITTER_CHUNKOVERLAP: 128
```

## Using a Custom Text Splitter

While the default text splitter works well, you can also implement a custom splitter for specific needs.

1. Modify the `get_text_splitter` method in `src/utils.py`.
   Update it to incorporate your custom text splitter class. Make sure to install its dependency in `requirements.txt` file.

   ```python
   def get_text_splitter():

      from langchain.text_splitter import RecursiveCharacterTextSplitter

      return RecursiveCharacterTextSplitter(
          chunk_size=get_config().text_splitter.chunk_size - 2,
          chunk_overlap=get_config().text_splitter.chunk_overlap
      )
   ```

   Make sure the chunks created by the function have a smaller number of tokens than the context length of the embedding model.

## Build and Start the Container

After you change the `get_text_splitter` function, build and start the container.
   ```console
   docker compose -f deploy/compose/docker-compose.yaml up -d --build
   ```


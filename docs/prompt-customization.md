<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Customize Prompts in NVIDIA RAG Blueprint

The [NVIDIA RAG Blueprint](readme.md) uses a [prompt.yaml](../src/nvidia_rag/rag_server/prompt.yaml) file that defines prompts for different contexts.
These prompts guide the RAG model in generating appropriate responses.
You can customize these prompts to fit your specific needs and achieve desired responses from the models.

## Default Prompts Overview

The `prompt.yaml` file contains a set of prompt templates used throughout the RAG system. Each prompt serves a specific purpose in the pipeline. Below is an overview of the available prompts and their usage:

### 1. `chat_template`
- **Purpose:** Used for general chat interactions with the assistant.
- **Usage:** Guides the assistant's tone and style for open-ended conversations. Used in the main chat pipeline when the `/generate` API is called with `use_knowledge_base: False`.

### 2. `rag_template`
- **Purpose:** Used for Retrieval-Augmented Generation (RAG) responses.
- **Usage:** Instructs the assistant to answer strictly based on provided context, with specific rules for conciseness and relevance. Used in the RAG pipeline when generating answers from retrieved documents, specifically when the `/generate` API is called with `use_knowledge_base: True`.

### 3. `query_rewriter_prompt`
- **Purpose:** Reformulates user questions into standalone queries.
- **Usage:** Ensures that follow-up questions referencing previous context are rewritten for clarity. Used by the [query rewriting component](./multiturn.md) before retrieval if enabled.

### 4. `reflection_relevance_check_prompt`
- **Purpose:** Evaluates the relevance of a context chunk to a user question.
- **Usage:** Assigns a relevance score (0, 1, or 2) to each context chunk. Used by the [reflection module](./self-reflection.md) for context filtering.

### 5. `reflection_query_rewriter_prompt`
- **Purpose:** Optimizes queries for high-precision vectorstore retrieval.
- **Usage:** Refines user questions for better semantic search performance. Used by the [reflection module](./self-reflection.md) during query rewriting.

### 6. `reflection_groundedness_check_prompt`
- **Purpose:** Checks if a generated response is grounded in the provided context.
- **Usage:** Assigns a groundedness score (0, 1, or 2) to the response. Used by the [reflection module](./self-reflection.md) for response validation.

### 7. `reflection_response_regeneration_prompt`
- **Purpose:** Regenerates a response to be more grounded in the context.
- **Usage:** Produces a new answer using only information supported by the context. Used by the [reflection module](./self-reflection.md) if the original response is not sufficiently grounded.

### 8. `document_summary_prompt`
- **Purpose:** Summarizes a document, preserving key metadata and findings.
- **Usage:** Generates concise summaries for ingested documents when they fit within a single chunk. Used if [document summarization](./summarization.md) is enabled during ingestion.
- **Context:** This prompt is used for initial summarization when the entire document content can be processed in one request.

### 9. `iterative_summary_prompt`
- **Purpose:** Updates an existing summary with new information from additional document chunks.
- **Usage:** Used for large documents that require chunked processing. Takes a previous summary and new chunk content to produce an updated comprehensive summary.
- **Context:** Part of the summarization strategies (single/hierarchical/iterative) described in [document summarization](./summarization.md). The iterative strategy (default, when `summarization_strategy` is `null` or omitted) uses this prompt for multi-chunk documents to maintain context across chunk boundaries.

### 10. `vlm_template`
- **Purpose:** Guides the assistant in answering questions using only provided images.
- **Usage:** Ensures answers are based solely on visual content, not external knowledge. Used in the [Vision-Language Model (VLM) pipeline](./vlm.md) if enabled.

### 11. `vlm_response_reasoning_template`
- **Purpose:** Decides whether a VLM response should be included in the final prompt for the LLM.
- **Usage:** Evaluates the relevance and usefulness of VLM responses. Used in [the VLM pipeline for response filtering](./vlm.md) if enabled.

### 12. `query_decomposition_multiquery_prompt`
- **Purpose:** Breaks down complex questions into simpler, self-contained subqueries.
- **Usage:** Decomposes complex questions into numbered, independent subqueries. Returns original query unchanged if already simple. Used in the [query decomposition pipeline](./query_decomposition.md).

### 13. `query_decompositions_query_rewriter_prompt`
- **Purpose:** Rewrites subsequent sub-queries using previously answered sub-query results for better contextual retrieval.
- **Usage:** Rewrites current sub-query using previous sub-query answers to make it more contextually aware for better document retrieval. Used in the [query decomposition pipeline](./query_decomposition.md).

### 14. `query_decomposition_followup_question_prompt`
- **Purpose:** Identifies missing information and generates follow-up questions when needed.
- **Usage:** Generates a single follow-up question if information is missing, or returns empty string if complete. Used in the [query decomposition pipeline](./query_decomposition.md).

### 15. `query_decomposition_final_response_prompt`
- **Purpose:** Generates the final response using conversation history and retrieved context.
- **Usage:** Generates final answers as "Envie" using only provided context, with strict grounding rules and no external knowledge. Used in the [query decomposition pipeline](./query_decomposition.md).

### 16. `shallow_summary_prompt`
- **Purpose:** Generates concise summaries for text-only (shallow) extraction workflows.
- **Usage:** Produces streamlined summaries when `shallow_summary: true` is set during document ingestion. Uses a simplified prompt optimized for fast text-only processing without multimodal elements (tables, images, charts).
- **Context:** Automatically selected when shallow extraction is enabled. For full multimodal extraction, `document_summary_prompt` is used instead. See [document summarization](./summarization.md) for details on shallow vs. full extraction.

### 17. `filter_expression_generator_prompt`
- **Purpose:** Converts natural language queries into precise metadata filter expressions for targeted document retrieval.
- **Usage:** Automatically generates filter expressions when `enable_filter_generator: true` is set in the `/generate` or `/chat/completions` API. The LLM analyzes the user's query and the collection's metadata schema to create appropriate filters.
- **Context:** This enables users to ask questions in natural language (e.g., "Show me Ford vehicles with infotainment features") and have the system automatically convert them into metadata filters (e.g., `content_metadata["manufacturer"] == "ford" AND array_contains(content_metadata["features"], "infotainment")`).
- **Customization:** For domain-specific applications, customize this prompt to map industry terminology to your metadata fields. See the [Customizing Filter Expression Generator Prompt](./custom-metadata.md#customizing-filter-expression-generator-prompt) section for detailed examples and best practices.

### 18. `image_captioning_prompt`
- **Purpose:** Generates detailed descriptions of images encountered during document ingestion.
- **Usage:** Used during the document ingestion process when image captioning is enabled. The prompt instructs the vision-language model to describe images in detail, including main subjects, actions, settings, and notable objects or features.
- **Context:** When documents containing images are ingested, this prompt is applied to generate textual descriptions that can be embedded and searched alongside the document text. The generated captions help improve retrieval accuracy for questions about visual content in the documents.

---

## Overriding Existing Templates in `prompt.yaml`

You can override any template defined in the packaged `prompt.yaml` by providing a custom prompt file and setting the `PROMPT_CONFIG_FILE` environment variable. When the service starts, it loads the default `prompt.yaml` (packaged in the container) and then merges it with your custom prompt file. If a key exists in both files, the value from your custom file will override the default.

**How the merging works:**
- Both the default and custom prompt files are loaded as Python dictionaries.
- The two dictionaries are merged.
- For any key present in both, the value from your custom file takes precedence.

### Example: Override `rag_template` with a Pirate Template

Suppose you want to make the RAG model respond as a pirate. You can override the `rag_template` by creating a custom prompt file like this:

```yaml
rag_template:
  human: |
    Arrr matey! Ye be speakin' to a pirate now. I answer all yer questions with the heart of a true buccaneer!
    Context: {context}
```

Save this as `/home/user/my_custom_prompt.yaml`.

**Set the environment variable:**
```bash
export PROMPT_CONFIG_FILE=/home/user/my_custom_prompt.yaml
```

**Restart the container (no rebuild needed):**
```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
```

Now, the service will use your pirate-themed `rag_template` instead of the default one. All other templates from the original `prompt.yaml` will remain unchanged unless you override them in your custom file.

---

**Tip:**
You can override multiple templates at once by specifying multiple keys in your custom prompt file. Any key not present in your custom file will fall back to the default.

## Prompt customization in Helm chart

The [prompt.yaml](../deploy/helm/nvidia-blueprint-rag/files/prompt.yaml) resides within the chart. This is converted into a ConfigMap within the Helm deployment.

To provide custom instructions in the prompt, you can edit the [prompt.yaml](../deploy/helm/nvidia-blueprint-rag/files/prompt.yaml) and update the `chat_template`, `rag_template` or `query_rewriter_prompt`.

```yaml
chat_template:
  human: |
    <custom prompt instructions>

rag_template:
  human: |
    <custom prompt instructions>

query_rewriter_prompt:
  human: |
    <custom prompt instructions>
```

After the required changes have been made, you can deploy the Helm chart from source by following the steps [here](deploy-helm-from-repo.md).


## Prompt Customization in Python Library Mode

When using the `nvidia_rag` Python package directly (library mode), you can customize prompts by passing them to the `NvidiaRAG` constructor. This is the recommended approach for library usage.

### Using a Dictionary

```python
from nvidia_rag import NvidiaRAG
from nvidia_rag.utils.configuration import NvidiaRAGConfig

# Define custom prompts as a dictionary
custom_prompts = {
    "rag_template": {
        "system": "/no_think",
        "human": """You are a helpful AI assistant named Envie.
You will reply to questions only based on the context that you are provided.
If something is out of context, you will refrain from replying and politely decline to respond to the user.

Context: {context}"""
    }
}

# Create NvidiaRAG instance with custom prompts
config = NvidiaRAGConfig.from_yaml("config.yaml")
rag = NvidiaRAG(config=config, prompts=custom_prompts)
```

### Using a YAML File

You can also load prompts from a YAML file:

```python
# Load prompts from a YAML file
rag = NvidiaRAG(config=config, prompts="custom_prompts.yaml")
```

For a complete working example with prompt customization in library mode, refer to the [RAG Library Usage Notebook](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/rag_library_usage.ipynb#9.-Customize-prompts).

---


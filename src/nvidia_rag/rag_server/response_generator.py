# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This module contains the response generator for the RAG server which generates the response to the user query and retrieves the summary of a document.
    1. response_generator(): Generate a response using the RAG chain.
    2. prepare_llm_request(): Prepare the request for the LLM response generation.
    3. generate_answer(): Generate and stream the response to the provided prompt.
    4. prepare_citations(): Prepare citations for the response.
    5. error_response_generator(): Generate a stream of data for the error response.
    6. retrieve_summary(): Retrieve the summary of a document.
"""

import logging
import time
import os
import bleach
import asyncio
from typing import Dict, Any, Generator, List, Union
from uuid import uuid4
from langchain_core.documents import Document
from pymilvus.exceptions import MilvusException, MilvusUnavailableException
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional

from nvidia_rag.utils.minio_operator import get_unique_thumbnail_id, get_minio_operator

logger = logging.getLogger(__name__)

FALLBACK_EXCEPTION_MSG = "Error from rag-server. Please check rag-server logs for more details."


class Usage(BaseModel):
    """Token usage information."""

    total_tokens: int = Field(
        default=0,
        ge=0,
        le=1000000000,
        format="int64",
        description="Total tokens used in the request",
    )
    prompt_tokens: int = Field(
        default=0,
        ge=0,
        le=1000000000,
        format="int64",
        description="Tokens used for the prompt",
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        le=1000000000,
        format="int64",
        description="Tokens used for the completion",
    )

class SourceMetadata(BaseModel):
    """Metadata associated with a document source."""

    language: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Language of the document",
    )
    date_created: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Creation date of the document",
    )
    last_modified: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Last modification date",
    )
    page_number: int = Field(
        default=0,
        ge=-1,
        le=1000000,
        format="int64",
        description="Page number in the document",
    )
    description: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Description of the document content",
    )
    height: int = Field(
        default=0,
        ge=0,
        le=100000,
        format="int64",
        description="Height of the document in pixels",
    )
    width: int = Field(
        default=0,
        ge=0,
        le=100000,
        format="int64",
        description="Width of the document in pixels",
    )
    location: List[float] = Field(
        default=[],
        description="Bounding box location of the content"
    )
    location_max_dimensions: List[int] = Field(
        default=[],
        description="Maximum dimensions of the document"
    )
    content_metadata: Dict[str, Any] = Field(
        default={},
        description="Metadata about the content"
    )

class SourceResult(BaseModel):
    """Represents a single source document result."""

    document_id: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Unique identifier of the document",
    )
    content: str = Field(
        default="",
        pattern=r"[\s\S]*",
        description="Extracted content from the document",
    )
    document_name: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Name of the document",
    )
    document_type: Literal["image", "text", "table", "chart", "audio"] = Field(
        default="text",
        description="Type of document content"
    )
    score: float = Field(
        default=0.0,
        description="Relevance score of the document")

    metadata: SourceMetadata

class Citations(BaseModel):
    """Represents the sources section of the API response."""

    total_results: int = Field(
        default=0,
        ge=0,
        le=1000000,
        format="int64",
        description="Total number of source documents found",
    )
    results: List[SourceResult] = Field(
        default=[], description="List of document results"
    )

class Message(BaseModel):
    """Definition of the Chat Message type."""

    role: Literal["user", "assistant", "system", None] = Field(
        description="Role for a message: either 'user' or 'assistant' or 'system",
        default="user"
    )
    content: str = Field(
        description="The input query/prompt to the pipeline.",
        default="Hello! What can you help me with?",
        max_length=131072,
        pattern=r'[\s\S]*',
    )

    @validator('role')
    @classmethod
    def validate_role(cls, value):
        """ Field validator function to validate values of the field role"""
        if value:
            value = bleach.clean(value, strip=True)
            valid_roles = {'user', 'assistant', 'system'}
            if value is not None and value.lower() not in valid_roles:
                raise ValueError("Role must be one of 'user', 'assistant', or 'system'")
            return value.lower()

    @validator('content')
    @classmethod
    def sanitize_content(cls, v):
        """ Feild validator function to santize user populated feilds from HTML"""
        return bleach.clean(v, strip=True)

class ChainResponseChoices(BaseModel):
    """ Definition of Chain response choices"""

    index: int = Field(default=0, ge=0, le=256, format="int64")
    message: Message = Field(default=Message(role="assistant", content=""))
    delta: Message = Field(default=Message(role=None, content=""))
    finish_reason: Optional[str] = Field(default=None, max_length=4096, pattern=r'[\s\S]*')

class ChainResponse(BaseModel):
    """Definition of Chain APIs resopnse data type"""

    id: str = Field(default="", max_length=100000, pattern=r'[\s\S]*')
    choices: List[ChainResponseChoices] = Field(default=[], max_items=256)
    # context will be deprecated once `sources` field is implemented and populated
    model: str = Field(default="", max_length=4096, pattern=r'[\s\S]*')
    object: str = Field(default="", max_length=4096, pattern=r'[\s\S]*')
    created: int = Field(default=0, ge=0, le=9999999999, format="int64")
    # Place holder fields for now to match generate API response structure
    usage: Optional[Usage] = Field(default=Usage(), description="Token usage statistics")
    citations: Optional[Citations] = Field(default=Citations(), description="Source documents used for the response")


def prepare_llm_request(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Prepare the request for the LLM response generation."""

    logger.debug(f"Prompt: {messages}")
    chat_history = [msg for msg in messages if not (msg.get('role') == 'assistant' and not msg.get('content', '').strip())]

    # Find the last user message and its index
    last_user_message = None
    last_user_index = None
    for i in range(len(chat_history) - 1, -1, -1):
        if chat_history[i].get('role') == 'user':
            last_user_message = chat_history[i].get('content')
            last_user_index = i
            break

    if last_user_message:
        last_user_message = escape_json_content(last_user_message)

    # Process chat history and escape JSON-like structures
    processed_chat_history = []
    for i, message in enumerate(chat_history):
        if i == last_user_index:
            # Skip only the last user message as it's handled separately
            continue
        # Create new Message with escaped content
        processed_message = {
            'role': message.get('role'),
            'content': escape_json_content(message.get('content', ''))
        }
        processed_chat_history.append(processed_message)

    logger.debug(f"User query: {last_user_message}, Chat history: {processed_chat_history}")
    return last_user_message, processed_chat_history


async def generate_answer(
    generator: 'Generator[str]',
    contexts: List[Any],
    model: str = "",
    collection_name: str = "",
    enable_citations: bool = True
):
    """Generate and stream the response to the provided prompt.

    Args:
        generator: Generator that yields response chunks
        contexts: List of context documents used for generation
        model: Name of the model used for generation
        collection_name: Name of the collection used for retrieval
        enable_citations: Whether to enable citations in the response
    """

    try:
        # unique response id for every query
        resp_id = str(uuid4())
        if generator:
            logger.debug("Generated response chunks\n")
            # Create ChainResponse object for every token generated
            first_chunk = True
            start_time = time.time()
            for chunk in generator:
                # TODO: This is a hack to clear contexts if we get an error response from nemoguardrails
                if chunk == "I'm sorry, I can't respond to that.":
                    # Clear contexts if we get an error response
                    contexts = list()
                chain_response = ChainResponse()
                response_choice = ChainResponseChoices(
                    index=0,
                    message=Message(role="assistant", content=chunk),
                    delta=Message(role=None, content=chunk),
                    finish_reason=None
                )
                chain_response.id = resp_id
                chain_response.choices.append(response_choice)  # pylint: disable=E1101
                chain_response.model = model
                chain_response.object = "chat.completion.chunk"
                chain_response.created = int(time.time())
                if first_chunk:
                    logger.info("    == LLM Time to First Token (TTFT): %.2f ms ==", (time.time() - start_time) * 1000)
                    chain_response.citations = prepare_citations(
                        retrieved_documents=contexts,
                        enable_citations=enable_citations,
                    )
                    first_chunk = False
                logger.debug(response_choice)
                # Send generator with tokens in ChainResponse format
                yield "data: " + str(chain_response.json()) + "\n\n"

            chain_response = ChainResponse()

            # [DONE] indicate end of response from server
            response_choice = ChainResponseChoices(
                finish_reason="stop",
            )
            chain_response.id = resp_id
            chain_response.choices.append(response_choice)  # pylint: disable=E1101
            chain_response.model = model
            chain_response.object = "chat.completion.chunk"
            chain_response.created = int(time.time())
            logger.debug(response_choice)
            yield "data: " + str(chain_response.json()) + "\n\n"
        else:
            chain_response = ChainResponse()
            yield "data: " + str(chain_response.json()) + "\n\n"

    except (MilvusException, MilvusUnavailableException) as e:
        exception_msg = ("Error from milvus server. Please ensure you have ingested some documents. "
                         "Please check rag-server logs for more details.")
        logger.error(
            "Error from Milvus database endpoint. Please ensure you have ingested some documents. " +
            "Error details: %s",
            e, exc_info=logger.getEffectiveLevel() <= logging.DEBUG)
        yield error_response_generator(exception_msg)

    except Exception as e:
        logger.error("Error from generate endpoint. Error details: %s", e,
                     exc_info=logger.getEffectiveLevel() <= logging.DEBUG)
        yield error_response_generator(FALLBACK_EXCEPTION_MSG)


def prepare_citations(
    retrieved_documents: List[Document],
    force_citations: bool = False, # True in-case of doc search api
    enable_citations: bool = True
) -> Citations:
    """
    Prepare citation information based on retrieved_documents
    Arguments:
        - collection_name: str - Milvus Collection Name
        - retrieved_documents: List of retrieved langchain documents
        - force_citations: This flag would give citations even if config enable_citations is unset
    Returns:
        - source_results: Citations
    """
    citations = list()

    if force_citations or enable_citations:
        for doc in retrieved_documents:

            if isinstance(doc.metadata.get("source"), str):
                # If langchain is used for ingestion, the source is a string
                file_name = os.path.basename(doc.metadata.get("source"))
                content = doc.page_content
                source_metadata = SourceMetadata(
                    description=doc.page_content,
                    content_metadata=doc.metadata.get("content_metadata", {})
                )
                document_type = "text"
            else:
                file_name = os.path.basename(doc.metadata.get("source").get("source_id"))

            if doc.metadata.get("content_metadata", {}).get("type") in ["text", "audio"]:
                content = doc.page_content
                document_type = doc.metadata.get("content_metadata", {}).get("type")
                source_metadata = SourceMetadata(
                    description=doc.page_content,
                    content_metadata=doc.metadata.get("content_metadata")
                )

            elif doc.metadata.get("content_metadata", {}).get("type", {}) in ["image", "structured"]:
                # Pull required metadata
                page_number = doc.metadata.get("content_metadata", {}).get("page_number")
                location = doc.metadata.get("content_metadata", {}).get("location")
                if doc.metadata.get("content_metadata", {}).get("type") == "image":
                    document_type = doc.metadata.get("content_metadata", {}).get("type")
                else:
                    document_type = doc.metadata.get("content_metadata", {}).get("subtype")
                try:
                    if enable_citations:
                        logger.debug("Pulling content from minio for image/table/chart for citations ...")
                        unique_thumbnail_id = get_unique_thumbnail_id(
                            collection_name=doc.metadata.get("collection_name"),
                            file_name=file_name,
                            page_number=page_number,
                            location=location
                        )
                        payload = get_minio_operator().get_payload(object_name=unique_thumbnail_id)
                        content = payload.get("content", "")
                        source_metadata = SourceMetadata(
                            page_number=page_number,
                            location=location,
                            description=doc.page_content,
                            content_metadata=doc.metadata.get("content_metadata")
                        )
                    else:
                        content = ""
                        source_metadata = SourceMetadata(
                            description=doc.page_content,
                            content_metadata=doc.metadata.get("content_metadata")
                        )
                except Exception as e:
                    logger.error(f"Error pulling content from minio for image/table/chart for citations: {e}")
                    content = ""
                    source_metadata = SourceMetadata(
                        description=doc.page_content,
                        content_metadata=doc.metadata.get("content_metadata", {})
                    )

            if content and document_type in ["image", "text", "table", "chart", "audio"]:
                # Prepare citations basemodel
                source_result = SourceResult(
                    content=content,
                    document_type=document_type,
                    document_name=file_name,
                    score=doc.metadata.get("relevance_score", 0),
                    metadata=source_metadata
                )
                citations.append(source_result)

    return Citations(
        total_results=len(citations),
        results=citations
    )


def error_response_generator(exception_msg: str):
    """
    Generate a stream of data for the error response
    """

    def get_chain_response(
        content: str = "",
        finish_reason: Union[str, None] = None
    ) -> ChainResponse:
        """
        Get a chain response for an exception
        Args:
            exception_msg: str - Exception message
        Returns:
            chain_response: ChainResponse - Chain response for an exception
        """
        chain_response = ChainResponse()
        chain_response.id = str(uuid4())
        response_choice = ChainResponseChoices(
            index=0,
            message=Message(role="assistant", content=content),
            delta=Message(role=None, content=content),
            finish_reason=finish_reason
        )
        chain_response.choices.append(response_choice)  # pylint: disable=E1101
        chain_response.object = "chat.completion.chunk"
        chain_response.created = int(time.time())
        return chain_response

    for i in range(0, len(exception_msg), 5):
        exception_msg_content = exception_msg[i:i+5]
        chain_response = get_chain_response(content=exception_msg_content)
        yield "data: " + str(chain_response.model_dump_json()) + "\n\n"
    chain_response = get_chain_response(finish_reason="stop")
    yield "data: " + str(chain_response.model_dump_json()) + "\n\n"


async def retrieve_summary(
        collection_name: str,
        file_name: str,
        wait: bool = False,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Get the summary of a document."""

        try:
            unique_thumbnail_id = get_unique_thumbnail_id(
                collection_name=f"summary_{collection_name}",
                file_name=file_name,
                page_number=0,
                location=[]
            )

            # First attempt to get existing summary
            payload = get_minio_operator().get_payload(object_name=unique_thumbnail_id)

            if payload:
                return {
                    "message": "Summary retrieved successfully.",
                    "summary": payload.get("summary", ""),
                    "file_name": payload.get("file_name", ""),
                    "collection_name": collection_name,
                    "status": "SUCCESS"
                }

            # If summary not found and wait=False, return immediately
            if not wait:
                return {
                    "message": f"Summary for {file_name} not found. Ensure the file name and collection name are correct. Set wait=true to wait for generation.",
                    "status": "FAILED"
                }

            # If wait=True, poll for summary with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                payload = get_minio_operator().get_payload(object_name=unique_thumbnail_id)
                if payload:
                    return {
                        "message": "Summary retrieved successfully.",
                        "summary": payload.get("summary", ""),
                        "file_name": payload.get("file_name", ""),
                        "collection_name": collection_name,
                        "status": "SUCCESS"
                    }

                # Wait before next poll
                await asyncio.sleep(2)

            # If timeout reached
            return {
                "message": f"Timeout waiting for summary generation for {file_name}",
                "status": "TIMEOUT"
            }

        except Exception as e:
            logger.error("Error from GET /summary endpoint. Error details: %s", e)
            return {
                "message": "Error occurred while getting summary.",
                "error": str(e),
                "status": "ERROR"
            }


# Helper function to escape JSON-like structures in content
def escape_json_content(content: str) -> str:
    """Escape curly braces in content to avoid JSON parsing issues"""
    return content.replace("{", "{{").replace("}", "}}")
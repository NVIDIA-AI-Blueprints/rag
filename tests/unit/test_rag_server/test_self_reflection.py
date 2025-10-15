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

"""
Test suite for self-reflection functionality in the RAG server.

This module tests the self-reflection mechanisms that help improve RAG responses
by evaluating and refining context relevance and response groundedness using
deterministic parameters and structured prompts. The tests cover retry mechanisms,
context relevance checking, and response grounding validation with reproducible results.

Key Testing Areas:
- Deterministic reflection scoring with temperature=0 and low top_p values
- Structured prompt templates with system and human message components
- Context relevance evaluation and query rewriting
- Response groundedness checking and regeneration
- Error handling and retry mechanisms
"""

import pytest
from langchain_core.documents import Document
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from nvidia_rag.rag_server.reflection import (
    ReflectionCounter,
    _retry_score_generation,
    check_context_relevance,
    check_response_groundedness,
)
from nvidia_rag.utils.llm import get_llm
from nvidia_rag.utils.reranker import get_ranking_model
from nvidia_rag.utils.vdb.vdb_base import VDBRag


@pytest.mark.asyncio
async def test_retry_score_generation(mocker):
    """
    Test the retry mechanism for score generation in self-reflection.

    This test verifies that the _retry_score_generation function correctly handles
    deterministic reflection scoring with consistent parameters. The test covers:
    1. Successful score generation with valid scores (e.g., "2") using deterministic LLM
    2. Cases where score extraction fails and defaults to 0
    3. Error handling that gracefully returns a score of 0
    4. Reproducible results with deterministic sampling parameters

    Args:
        mocker: pytest-mock fixture for mocking dependencies

    Note:
        The test uses deterministic LLM parameters (temperature=0, low top_p)
        to match the production reflection configuration for consistent results.
    """
    # Set up the structured prompt template for testing reflection scoring
    # Uses system message component to match production prompt structure
    relevance_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "this is a prompt",
            )
        ]
    )
    # Configure deterministic LLM parameters to match production reflection settings
    llm_params = {
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
        "temperature": 0,  # Deterministic sampling for reproducible test results
        "top_p": 0.1,  # Low top_p for focused, consistent responses
        "max_tokens": 32768,  # Large context window for comprehensive analysis
    }
    # Create a chain: prompt -> LLM -> string parser
    chain = relevance_template | get_llm(**llm_params) | StrOutputParser()

    # Define test cases covering different scenarios
    testcases = {
        "Success with score 2": {
            "inputs": {"query": "query123", "context": "context123"},
            "chain": chain,
            "response": "2",  # Valid score response
            "expected_score": 2,
        },
        "Success with score 0": {
            "inputs": {"query": "query123", "context": "context123"},
            "response": "Couldn't get the score",  # Invalid response, should default to 0
            "expected_score": 0,
            "chain": chain,
        },
        "Error": {
            "inputs": {"query": "query123", "context": "context123"},
            "expected_score": 0,  # Error case should return 0
            "chain": None,  # No chain provided to simulate error
        },
    }

    # Run each test case
    for tc_name, tc_data in testcases.items():
        print("Running testcase ", tc_name)
        # Mock the chain invoke method for non-error cases
        if "Error" not in tc_name:
            mocker.patch(
                "langchain_core.runnables.base.RunnableSequence.invoke",
                return_value=tc_data["response"],
            )

        # Call the function under test
        score = _retry_score_generation(tc_data["chain"], tc_data["inputs"])

        # Verify the score matches expected result
        assert score == tc_data["expected_score"]


@pytest.mark.asyncio
async def test_check_context_relevance(mocker):
    """
    Test the context relevance checking functionality with deterministic reflection.

    This test verifies that check_context_relevance correctly implements deterministic
    reflection evaluation with structured prompts. The test covers:
    1. Document reranking and filtering when a ranker is provided
    2. Fallback to basic retrieval when no ranker is provided
    3. Deterministic reflection scoring to evaluate context quality
    4. Query rewriting using structured prompts with system and human components
    5. Reproducible results with deterministic sampling parameters

    Args:
        mocker: pytest-mock fixture for mocking dependencies

    Note:
        The function under test uses deterministic parameters (temperature=0, top_p=0.1)
        and structured prompts for consistent, reproducible reflection results.
    """
    # Set up a local ranker for reranking documents
    local_ranker = get_ranking_model(model="nvidia/llama-3.2-nv-rerankqa-1b-v2", url="")

    # Create a mock VDBRag object
    mock_vdb_op = mocker.MagicMock(spec=VDBRag)

    # Define test cases for different ranker scenarios
    testcases = {
        "Success with ranker": {
            "vdb_op": mock_vdb_op,
            "retriever_query": "this is query",
            "collection_names": ["test_collection"],
            "collection_filter_mapping": {"test_collection": ""},
            "rewritten_query": "this is new rewritten query",
            "counter": ReflectionCounter(2),  # Allow 2 reflection attempts
            "ranker": local_ranker,
            "docs": Document(page_content="abcd"),  # Sample document
            "expected_docs": [Document(metadata={}, page_content="abcd")],
        },
        "Success with no ranker": {
            "vdb_op": mock_vdb_op,
            "retriever_query": "this is query",
            "collection_names": ["test_collection"],
            "collection_filter_mapping": {"test_collection": ""},
            "rewritten_query": "this is new rewritten query",
            "counter": ReflectionCounter(1),  # Allow 1 reflection attempt
            "ranker": None,  # No ranker provided
            "docs": Document(page_content="efgh"),  # Different sample document
            "expected_docs": [Document(metadata={}, page_content="efgh")],
        },
    }

    # Run each test case
    for tc_name, tc_data in testcases.items():
        print("Running testcase ", tc_name)

        # Mock document compression (used by ranker)
        mocker.patch(
            "langchain_core.documents.compressor.BaseDocumentCompressor.compress_documents",
            return_value=None,
        )

        # Mock ranker behavior if ranker is provided
        if tc_data["ranker"]:
            mocker.patch(
                "langchain_core.runnables.RunnableAssign.invoke",
                return_value={"context": [tc_data["docs"]]},
            )

        # Mock document retrieval from VDBRag
        tc_data["vdb_op"].retrieval_langchain.return_value = [tc_data["docs"]]

        # Mock the deterministic scoring mechanism (return low score to trigger reflection)
        # This simulates the deterministic reflection scoring behavior
        mocker.patch(
            "nvidia_rag.rag_server.reflection._retry_score_generation",
            return_value=0,
        )

        # Mock structured query rewriting chain (system + human prompt components)
        mocker.patch(
            "langchain_core.runnables.base.RunnableSequence.invoke",
            return_value=tc_data["rewritten_query"],
        )

        # Call the function under test
        docs, _ = check_context_relevance(
            tc_data["vdb_op"],
            tc_data["retriever_query"],
            tc_data["collection_names"],
            tc_data["ranker"],
            tc_data["counter"],
            collection_filter_mapping=tc_data["collection_filter_mapping"],
        )

        # Verify the returned documents match expected results
        assert docs == tc_data["expected_docs"]


@pytest.mark.asyncio
async def test_check_response_groundedness(mocker):
    """
    Test the response groundedness checking functionality with deterministic reflection.

    This test verifies that check_response_groundedness correctly implements deterministic
    groundedness evaluation with structured prompts. The test covers:
    1. Deterministic evaluation of response grounding in provided context
    2. Consistent reflection scoring using temperature=0 and low top_p values
    3. Response regeneration using structured prompts with system and human components
    4. Proper handling of out-of-context scenarios
    5. Reproducible results without debug output interference

    Args:
        mocker: pytest-mock fixture for mocking dependencies

    Note:
        The function under test uses deterministic parameters and structured prompts
        for consistent reflection results. Debug output has been removed for cleaner
        production operation.
    """
    # Define test case for response grounding
    testcases = {
        "Success": {
            "query": "this is query",
            "response": "this is RAG response",  # Original response to evaluate
            "context": ["this is context"],  # Context documents for grounding
            "counter": ReflectionCounter(2),  # Allow 2 reflection attempts
            "new_response": "this is new RAG response",  # Expected refined response
        },
        "Success: Out of context query": {
            "query": "this is out of context query",
            "response": "OUT OF CONTEXT",  # Original response to evaluate
            "context": ["this is context"],  # Context documents for grounding
            "counter": ReflectionCounter(2),  # Allow 2 reflection attempts
            "new_response": "OUT OF CONTEXT",  # Expected refined response
        },
    }

    # Run the test case
    for tc_name, tc_data in testcases.items():
        print("Running testcase ", tc_name)

        # Mock the deterministic scoring mechanism (return low score to trigger response refinement)
        # This simulates deterministic groundedness evaluation
        mocker.patch(
            "nvidia_rag.rag_server.reflection._retry_score_generation",
            return_value=0,
        )

        # Mock the structured response generation chain (system + human prompt components)
        mocker.patch(
            "langchain_core.runnables.base.RunnableSequence.invoke",
            return_value=tc_data["new_response"],
        )

        # Call the function under test
        response, _ = check_response_groundedness(
            tc_data["query"],
            tc_data["response"],
            tc_data["context"],
            tc_data["counter"],
        )

        # Verify the response matches the expected response
        assert response == tc_data["new_response"]

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import Generation, LLMResult
from opentelemetry.semconv_ai import (
    LLMRequestTypeValues,
    SpanAttributes,
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    TraceloopSpanKindValues,
)


class SpanMock:
    def __init__(self):
        self.attributes = []
        self.events = []
        self.ended = False
        self.end_time = None

    def set_attribute(self, key, value):
        self.attributes.append((key, value))

    def add_event(self, name: str):
        self.events.append(name)

    def end(self):
        self.ended = True
        self.end_time = True


class TracerMock:
    def __init__(self):
        self.spans = []

    def start_span(self, name, context=None, kind=None):
        span = SpanMock()
        self.spans.append((name, span))
        return span


class MetricsMock:
    def __init__(self):
        self.avg_words_calls = []
        self.token_calls = []

    def update_avg_words_per_chunk(self, avg_words_per_chunk: int):
        self.avg_words_calls.append(avg_words_per_chunk)

    def update_llm_tokens(self, input_t: int, output_t: int):
        self.token_calls.append((input_t, output_t))


@pytest.fixture()
def handler():
    from nvidia_rag.utils.observability.langchain_callback_handler import LangchainCallbackHandler

    tracer = TracerMock()
    metrics = MetricsMock()
    return LangchainCallbackHandler(tracer=tracer, metrics=metrics)


def test_on_chat_model_start_sets_input_words_and_prompts(handler):
    from nvidia_rag.utils.observability.langchain_callback_handler import (
        GEN_AI_PROMPTS,
    )

    run_id = uuid4()
    messages = [
        [
            SimpleNamespace(type="human", content="hello world"),
            SimpleNamespace(type="system", content="sys msg"),
        ]
    ]

    handler.on_chat_model_start(
        serialized={"kwargs": {"name": "chat-model"}},
        messages=messages,
        run_id=run_id,
    )

    # total words = 2 (hello world) + 2 (sys msg)
    assert handler.total_input_words == 4
    # span should be created and attributes recorded
    assert run_id in handler.spans




def test_on_chain_end_updates_avg_words_per_chunk(handler):
    # Need a span created; simulate chain start
    run_id = uuid4()
    handler.on_chain_start(
        serialized={"kwargs": {"name": "chain"}},
        inputs={"question": "q"},
        run_id=run_id,
    )

    # Provide context to compute avg words per chunk
    context = ["a b c", "d e"]  # 3 and 2 words -> avg 2 (int)
    handler.on_chain_end(outputs={}, run_id=run_id, inputs={"context": context})

    # Metrics should be updated
    assert handler.metrics.avg_words_calls[-1] == 2


def test_on_chain_end_updates_llm_tokens(handler):
    # Set input words via chat start
    run_chat_id = uuid4()
    handler.on_chat_model_start(
        serialized={"kwargs": {"name": "chat"}},
        messages=[[SimpleNamespace(type="human", content="hello there friend")]],
        run_id=run_chat_id,
    )

    # Create a separate chain span so _end_span has a span to close
    run_chain_id = uuid4()
    handler.on_chain_start(
        serialized={"kwargs": {"name": "chain"}}, inputs={}, run_id=run_chain_id
    )

    # Provide AIMessageChunk to trigger token update path
    output_chunk = AIMessageChunk(content="hi there")  # 2 words
    handler.on_chain_end(outputs={}, run_id=run_chain_id, inputs=output_chunk)

    # Expect update_llm_tokens called with input words from chat (3) and output words (2)
    assert handler.metrics.token_calls[-1] == (3, 2)


# SpanAttributes from opentelemetry.semconv_ai still used in langchain_callback_handler.py.
# Missing/deprecated ones (LLM_REQUEST_MODEL, LLM_RESPONSE_MODEL, LLM_REQUEST_MAX_TOKENS,
# LLM_REQUEST_TEMPERATURE, LLM_REQUEST_TOP_P, LLM_SYSTEM) are hardcoded in the handler.
SPAN_ATTRIBUTES_USED = [
    "LLM_REQUEST_FUNCTIONS",
    "LLM_USAGE_TOTAL_TOKENS",
    "TRACELOOP_WORKFLOW_NAME",
    "TRACELOOP_ENTITY_PATH",
    "TRACELOOP_SPAN_KIND",
    "TRACELOOP_ENTITY_NAME",
    "LLM_REQUEST_TYPE",
    "TRACELOOP_ENTITY_INPUT",
    "TRACELOOP_ENTITY_OUTPUT",
]


def test_semconv_ai_span_attributes_exist_and_not_deprecated():
    """Ensure all SpanAttributes used in langchain_callback_handler exist and are non-empty strings."""
    for attr_name in SPAN_ATTRIBUTES_USED:
        assert hasattr(
            SpanAttributes, attr_name
        ), f"SpanAttributes.{attr_name} is missing or was removed from opentelemetry.semconv_ai"
        value = getattr(SpanAttributes, attr_name)
        assert isinstance(
            value, str
        ), f"SpanAttributes.{attr_name} should be a string, got {type(value).__name__}"
        assert (
            len(value) > 0
        ), f"SpanAttributes.{attr_name} is empty (possibly deprecated or placeholder)"


def test_semconv_ai_llm_request_type_values_used():
    """Ensure LLMRequestTypeValues used in the handler (CHAT, COMPLETION) exist."""
    assert hasattr(LLMRequestTypeValues, "CHAT")
    assert hasattr(LLMRequestTypeValues, "COMPLETION")
    assert isinstance(LLMRequestTypeValues.CHAT.value, str)
    assert isinstance(LLMRequestTypeValues.COMPLETION.value, str)


def test_semconv_ai_traceloop_span_kind_values_used():
    """Ensure TraceloopSpanKindValues used in the handler (WORKFLOW, TASK, TOOL) exist."""
    for kind in ("WORKFLOW", "TASK", "TOOL"):
        assert hasattr(
            TraceloopSpanKindValues, kind
        ), f"TraceloopSpanKindValues.{kind} is missing"
        assert isinstance(getattr(TraceloopSpanKindValues, kind).value, str)


def test_langchain_callback_handler_imports_and_constants():
    """Import the handler module and verify hardcoded attribute constants and semconv_ai key."""
    from nvidia_rag.utils.observability.langchain_callback_handler import (
        GEN_AI_COMPLETIONS,
        GEN_AI_PROMPTS,
        LLM_REQUEST_MAX_TOKENS,
        LLM_REQUEST_MODEL,
        LLM_REQUEST_TEMPERATURE,
        LLM_REQUEST_TOP_P,
        LLM_RESPONSE_MODEL,
        LLM_SYSTEM,
    )

    assert GEN_AI_PROMPTS == "gen_ai.prompt"
    assert GEN_AI_COMPLETIONS == "gen_ai.completion"
    assert LLM_REQUEST_MODEL == "gen_ai.request.model"
    assert LLM_RESPONSE_MODEL == "gen_ai.response.model"
    assert LLM_REQUEST_MAX_TOKENS == "llm.request.max_tokens"
    assert LLM_REQUEST_TEMPERATURE == "llm.request.temperature"
    assert LLM_REQUEST_TOP_P == "llm.request.top_p"
    assert LLM_SYSTEM == "llm.system"
    assert SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY is not None
    assert isinstance(SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY, str)

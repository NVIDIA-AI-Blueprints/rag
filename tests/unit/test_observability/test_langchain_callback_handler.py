# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import Generation, LLMResult


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
    from observability.langchain_callback_handler import LangchainCallbackHandler

    tracer = TracerMock()
    metrics = MetricsMock()
    return LangchainCallbackHandler(tracer=tracer, metrics=metrics)


def test_on_chat_model_start_sets_input_words_and_prompts(handler):
    from observability.langchain_callback_handler import SpanAttributes

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
    span = handler.spans[run_id].span
    # Check that at least one prompt attribute key prefix was used
    prompt_prefix = f"{SpanAttributes.LLM_PROMPTS}."
    prompt_keys = [k for k, _ in span.attributes if k.startswith(prompt_prefix)]
    assert len(prompt_keys) >= 2


def test_on_llm_start_and_end_sets_token_usage_and_ends_span(handler):
    run_id = uuid4()

    handler.on_llm_start(
        serialized={"kwargs": {"name": "llm"}},
        prompts=["What is ML?"],
        run_id=run_id,
    )

    # Build a minimal valid LLMResult
    gen = Generation(
        text="Answer",
        generation_info={"finish_reason": "stop"},
    )

    llm_result = LLMResult(
        generations=[[gen]],
        llm_output={
            "model_name": "test-model",
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        },
    )

    handler.on_llm_end(response=llm_result, run_id=run_id)

    span = handler.spans[run_id].span
    # Verify span ended
    assert span.ended is True
    # Verify some token usage attributes were set from llm_output usage
    attr_keys = [k for k, _ in span.attributes]
    assert any("usage" in k.lower() for k in attr_keys)


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

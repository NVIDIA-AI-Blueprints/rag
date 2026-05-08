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
Pydantic v2 configuration models for rag-perf, plus the three enums that
classify input sources and load-generation modes.

Composition follows the YAML structure: ``RunConfig`` is the top-level model
that holds the nested ``target``, ``aiperf``, ``load``, ``rag``, ``generation``,
``input``, ``output``, and ``sweep`` sub-configs. ``RunConfig.from_yaml(path)``
is the canonical loader used by the CLI.

Behaviour is fully config-driven; there are no separate "modes" or subcommands:

- ``load.concurrency`` accepts either an int (single point) or a list (sweep
  across that axis).
- ``aiperf.enabled = false`` skips the load-test phase, leaving only the
  server-side profiling pass.
- ``sweep.{vdb_top_k, reranker_top_k}`` are optional cross-axes; when set,
  the run becomes a Cartesian-product sweep.
"""

from __future__ import annotations

import io
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from ruamel.yaml import YAML


class InputSource(str, Enum):
    """How queries are sourced for the benchmark run."""

    JSONL = "jsonl"
    """Load queries from a .jsonl file — one JSON object per line with a ``query`` key."""

    CSV = "csv"
    """Load queries from a CSV file that must contain at least a ``query`` column."""

    SYNTHETIC = "synthetic"
    """
    Generate synthetic queries via an LLM endpoint.

    Two modes are supported (``input.synthetic.mode``):
    - ``random``       — LLM generates queries from scratch.
    - ``dataset_based``— LLM generates queries inspired by reference questions
                         from a dataset file (``input.synthetic.dataset_file``
                         or ``input.synthetic.dataset_name``).

    Generated queries are written to ``input.synthetic.jsonl_output_path``
    before being used, so the downstream pipeline always reads from a JSONL.
    """


class SyntheticMode(str, Enum):
    """LLM-based synthetic query generation strategy."""

    RANDOM = "random"
    """LLM generates queries from scratch with no reference material."""

    DATASET_BASED = "dataset_based"
    """LLM generates queries inspired by reference questions from a dataset file."""


class LoadMode(str, Enum):
    """The load-generation strategy passed to aiperf."""

    CONCURRENCY = "concurrency"
    """Keep exactly N requests in-flight at all times (fixed concurrency)."""

    REQUEST_RATE = "request_rate"
    """Issue requests at N req/s using Poisson inter-arrival times."""


class TargetConfig(BaseModel):
    """Where the NVIDIA RAG Blueprint server lives."""

    url: str = Field(
        default="http://localhost:8081",
        description="Base URL of the RAG server (no trailing slash).",
    )
    timeout_s: int = Field(
        default=300,
        ge=1,
        description="Per-request wall-clock timeout in seconds.",
    )


class RagParams(BaseModel):
    """
    RAG-specific parameters forwarded verbatim to the RAG server's
    ``POST /v1/generate`` API on every request.

    Any field here can be overridden per-query by including it in the query's
    JSONL entry (see ``examples/queries.jsonl``).
    """

    collection_names: list[str] = Field(
        default=["default"],
        description=(
            "Vector DB collection(s) to search. "
            "Maps to the RAG server's ``collection_names`` field."
        ),
    )
    vdb_top_k: int | list[int] = Field(
        default=100,
        description=(
            "Chunks retrieved from the vector DB before reranking. "
            "Scalar (e.g. ``100``) → single value; list (e.g. ``[20, 100]``) → "
            "sweep across the listed values. Maps to ``vdb_top_k`` per request."
        ),
    )
    reranker_top_k: int | list[int] = Field(
        default=10,
        description=(
            "Chunks passed to the LLM after reranking. "
            "Scalar (e.g. ``10``) → single value; list (e.g. ``[4, 10]``) → "
            "sweep across the listed values. Maps to ``reranker_top_k`` per request."
        ),
    )
    enable_reranker: bool = Field(
        default=True,
        description="Whether to run the reranker stage. Maps to ``enable_reranker``.",
    )
    enable_citations: bool = Field(
        default=True,
        description=(
            "Whether the server returns citation chunks in the response. "
            "Maps to ``enable_citations``."
        ),
    )
    use_knowledge_base: bool = Field(
        default=True,
        description=(
            "Set False to skip retrieval and send the bare query directly to the LLM. "
            "Maps to ``use_knowledge_base``."
        ),
    )
    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum relevance score for a retrieved chunk to be included. "
            "Maps to ``confidence_threshold``."
        ),
    )
    fetch_full_page_context: bool = Field(
        default=False,
        description=(
            "Fetch all chunks from each retrieved page (PDF-centric use cases). "
            "Maps to ``fetch_full_page_context``."
        ),
    )

    @model_validator(mode="after")
    def _validate_top_k_axes(self) -> RagParams:
        for name, value, lo, hi in (
            ("vdb_top_k", self.vdb_top_k, 1, 400),
            ("reranker_top_k", self.reranker_top_k, 1, 25),
        ):
            values = value if isinstance(value, list) else [value]
            if not values:
                raise ValueError(
                    f"rag.{name} must be a positive int or a non-empty list of ints"
                )
            for v in values:
                if v < lo or v > hi:
                    raise ValueError(
                        f"rag.{name} entries must be in [{lo}, {hi}]; got {v}"
                    )
            if isinstance(value, list) and len(set(values)) != len(values):
                raise ValueError(
                    f"rag.{name} must not contain duplicate values; got {values}. "
                    f"Each value would map to the same point directory and overwrite the previous."
                )
        return self

    @property
    def vdb_top_k_list(self) -> list[int]:
        """``vdb_top_k`` normalised to a list (scalar → single-element list)."""
        return list(self.vdb_top_k) if isinstance(self.vdb_top_k, list) else [self.vdb_top_k]

    @property
    def reranker_top_k_list(self) -> list[int]:
        """``reranker_top_k`` normalised to a list (scalar → single-element list)."""
        return (
            list(self.reranker_top_k)
            if isinstance(self.reranker_top_k, list)
            else [self.reranker_top_k]
        )


class GenerationParams(BaseModel):
    """LLM generation parameters forwarded to the RAG server."""

    max_tokens: int = Field(
        default=512,
        ge=1,
        description="Maximum output tokens. Maps to ``max_tokens``.",
    )
    min_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Minimum output tokens. When set, forces the model to generate at least "
            "this many tokens. Set equal to ``max_tokens`` to pin output length exactly "
            "(equivalent to the blueprint pipeline's ``min_tokens:$OSL`` extra-input)."
        ),
    )
    ignore_eos: bool = Field(
        default=False,
        description=(
            "Pass ``ignore_eos:true`` to the inference backend, preventing early "
            "stop on EOS tokens. Use together with ``min_tokens`` to enforce a fixed "
            "output length regardless of content (mirrors ``ignore_eos:true`` in the "
            "blueprint pipeline)."
        ),
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Maps to ``temperature``.",
    )


class LoadConfig(BaseModel):
    """Load-generation parameters controlling how aiperf sends requests.

    ``concurrency`` accepts either a scalar int (single benchmark point) or a
    list of ints (sweep across that axis). A 1-element list is equivalent to
    a scalar.
    """

    mode: LoadMode = Field(
        default=LoadMode.CONCURRENCY,
        description=(
            "``concurrency``: N workers always active. "
            "``request_rate``: Poisson arrivals at N req/s."
        ),
    )
    concurrency: int | list[int] = Field(
        default=8,
        description=(
            "Concurrent in-flight requests. Scalar (e.g. ``8``) → single "
            "benchmark point. List (e.g. ``[1, 4, 8, 16]``) → sweep across "
            "the listed values."
        ),
    )
    request_rate: float | None = Field(
        default=None,
        gt=0.0,
        description="Target requests per second (request_rate mode only).",
    )
    warmup_requests: int = Field(
        default=10,
        ge=1,
        description=(
            "Requests sent (and discarded) before measurement starts. "
            "Must be >= 1: aiperf rejects warmup=0."
        ),
    )
    total_requests: int = Field(
        default=200,
        ge=1,
        description="Total measured requests per point (excluding warmup).",
    )
    duration_s: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "If set, run for this many wall-clock seconds instead of "
            "``total_requests``. Cannot be combined with ``total_requests``."
        ),
    )
    profile_requests: int = Field(
        default=20,
        ge=1,
        description=(
            "Number of requests in the server-side profiling pass that runs "
            "before the aiperf load test. The profiling pass uses direct "
            "httpx calls to capture per-stage timing (retrieval, reranking, "
            "LLM TTFT) that aiperf cannot see."
        ),
    )
    iterations: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of times to repeat the full grid (concurrency × vdb_top_k "
            "× reranker_top_k). Each repetition writes to its own ``iter_<i>/`` "
            "subdirectory so results can be averaged or compared across runs "
            "to assess variance."
        ),
    )
    sleep_between_points_s: int = Field(
        default=0,
        ge=0,
        description=(
            "Seconds to sleep between grid points. Allows the server to drain "
            "in-flight requests and return to idle before the next measurement "
            "starts. Set to 60 to match the blueprint pipeline's SLEEP_TIME default."
        ),
    )

    @model_validator(mode="after")
    def _validate_concurrency(self) -> LoadConfig:
        cs = self.concurrency_list
        if not cs:
            raise ValueError(
                "load.concurrency must be a positive int or a non-empty list of positive ints"
            )
        for c in cs:
            if c < 1:
                raise ValueError(
                    f"load.concurrency entries must be >= 1; got {c}"
                )
        if len(set(cs)) != len(cs):
            raise ValueError(
                f"load.concurrency must not contain duplicate values; got {cs}. "
                f"Each value would map to the same point directory and overwrite the previous."
            )
        return self

    @model_validator(mode="after")
    def _validate_rate_mode(self) -> LoadConfig:
        if self.mode == LoadMode.REQUEST_RATE and self.request_rate is None:
            raise ValueError(
                "load.request_rate must be set when load.mode='request_rate'"
            )
        return self

    @property
    def concurrency_list(self) -> list[int]:
        """Concurrency normalised to a list (scalar → single-element list)."""
        return (
            list(self.concurrency)
            if isinstance(self.concurrency, list)
            else [self.concurrency]
        )


class AiperfConfig(BaseModel):
    """Whether to drive the aiperf load-test phase after profiling."""

    enabled: bool = Field(
        default=True,
        description=(
            "When True (default), each benchmark point runs the server-side "
            "profiling pass *and* the aiperf load test. When False, only the "
            "profiling pass runs — useful for fast iteration on retrieval / "
            "reranker tuning without triggering full load generation."
        ),
    )


class SyntheticInputConfig(BaseModel):
    """Parameters for LLM-based synthetic query generation."""

    mode: SyntheticMode = Field(
        default=SyntheticMode.RANDOM,
        description=(
            "``random``       — LLM generates queries from scratch.\n"
            "``dataset_based`` — LLM generates queries inspired by reference questions."
        ),
    )
    num_queries: int = Field(
        default=50,
        ge=1,
        description=(
            "Number of distinct synthetic queries to generate. "
            "The query list is cycled automatically if total_requests exceeds this."
        ),
    )
    min_query_tokens: int = Field(
        default=50,
        ge=1,
        description="Approximate minimum token count the LLM should target per generated query.",
    )
    generation_concurrency: int = Field(
        default=8,
        ge=1,
        description=(
            "Maximum concurrent LLM calls during synthetic generation. Bounds "
            "thread fan-out; raise to speed up generation against fast endpoints, "
            "lower to be polite to slow / rate-limited ones."
        ),
    )
    temperature: float = Field(
        default=0.9,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the synthetic-query LLM call.",
    )
    disable_thinking: bool = Field(
        default=True,
        description=(
            "When True, inject ``chat_template_kwargs: {enable_thinking: false}`` "
            "into the LLM payload. Reasoning models (Nemotron Omni, "
            "Qwen2.5-Reasoning, etc.) otherwise spend the token budget on chain-"
            "of-thought and leave the ``content`` field empty or polluted with "
            "deliberation text. Disable only if your endpoint is non-reasoning "
            "or rejects this kwarg."
        ),
    )
    extra_body: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Escape hatch: arbitrary keys merged into the LLM request body "
            "(e.g. ``top_p``, ``presence_penalty``, ``response_format``). "
            "Merged after ``disable_thinking`` so explicit keys here win."
        ),
    )
    llm_url: str = Field(
        default="http://localhost:8999/v1/chat/completions",
        description="OpenAI-compatible chat completions endpoint used for query generation.",
    )
    llm_model: str = Field(
        default="",
        description=(
            "Model name passed to the LLM endpoint. "
            "If empty, auto-discovered from the endpoint's /v1/models."
        ),
    )
    prompts_file: str | None = Field(
        default=None,
        description=(
            "Path to the YAML file containing prompt templates for synthetic generation. "
            "If not set, ``prompts/default_prompts.yaml`` is used when it exists, "
            "otherwise the prompts bundled with the package are used."
        ),
    )
    jsonl_output_path: str = Field(
        default="./rag-perf-synthetic-queries.jsonl",
        description=(
            "File path where generated synthetic queries are written as JSONL before "
            "being consumed.  The file is (over-)written on every generation run."
        ),
    )
    dataset_file: str | None = Field(
        default=None,
        description=(
            "Explicit path to a JSON/JSONL dataset file whose ``question`` entries "
            "are used as reference seeds (required for ``dataset_based`` mode when "
            "``dataset_name`` is not set)."
        ),
    )
    dataset_name: str | None = Field(
        default=None,
        description=(
            "Dataset name for auto-lookup under ``./datasets/``.  Alternative to "
            "``dataset_file`` for ``dataset_based`` mode.  The application searches "
            "for ``./datasets/<name>/train.json``, ``./datasets/<name>.json``, and "
            "``./datasets/<name>/data.json`` in order."
        ),
    )

    @model_validator(mode="after")
    def _validate_dataset_based(self) -> SyntheticInputConfig:
        if (
            self.mode == SyntheticMode.DATASET_BASED
            and self.dataset_file is None
            and self.dataset_name is None
        ):
            raise ValueError(
                "For synthetic.mode='dataset_based', set either "
                "synthetic.dataset_file (explicit path) or "
                "synthetic.dataset_name (auto-lookup under ./datasets/)."
            )
        return self


class InputConfig(BaseModel):
    """How query inputs are loaded and sampled.

    Exactly one of ``file`` or ``synthetic`` must be set; they are mutually
    exclusive. The format of ``file`` is auto-detected from its extension
    (``.jsonl`` or ``.csv``).
    """

    file: str | None = Field(
        default=None,
        description=(
            "Path to a query file. Format auto-detected from extension:\n\n"
            "* ``.jsonl`` — one JSON object per line with a ``query`` key.\n"
            "* ``.csv``   — must have a ``query`` column.\n\n"
            "Per-query overrides: any field from ``RagParams`` or "
            "``GenerationParams`` may be included alongside ``query``.\n\n"
            "Mutually exclusive with ``synthetic``."
        ),
    )
    synthetic: SyntheticInputConfig | None = Field(
        default=None,
        description=(
            "Generate synthetic queries via an LLM endpoint. "
            "Mutually exclusive with ``file``."
        ),
    )
    sampling: str = Field(
        default="random",
        description=(
            "How queries are selected when ``total_requests > len(queries)``:\n"
            "  ``random``       — random with replacement (default)\n"
            "  ``sequential``   — cycle through in order\n"
            "  ``shuffle-once`` — shuffle once, then cycle"
        ),
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducible sampling.",
    )

    @model_validator(mode="after")
    def _validate_xor(self) -> InputConfig:
        has_file = self.file is not None
        has_synthetic = self.synthetic is not None
        if has_file and has_synthetic:
            raise ValueError(
                "input.file and input.synthetic are mutually exclusive; set one, not both."
            )
        if not has_file and not has_synthetic:
            # Default to synthetic mode (with all-default SyntheticInputConfig) so
            # RunConfig.defaults() and bare YAML files still validate.
            self.synthetic = SyntheticInputConfig()
        if has_file:
            ext = Path(self.file).suffix.lower()
            if ext not in (".jsonl", ".csv"):
                raise ValueError(
                    f"input.file must end in .jsonl or .csv (extension determines format); "
                    f"got {self.file!r} (extension {ext!r})."
                )
        return self

    @property
    def source(self) -> InputSource:
        """Derived input source — for internal dispatch in QueryLoader."""
        if self.synthetic is not None:
            return InputSource.SYNTHETIC
        ext = Path(self.file).suffix.lower()  # type: ignore[arg-type]
        return InputSource.CSV if ext == ".csv" else InputSource.JSONL


class OutputConfig(BaseModel):
    """Where and how benchmark results are persisted."""

    dir: str = Field(
        default="./rag-perf-results",
        description="Root output directory. A timestamped sub-directory is created per run.",
    )
    formats: list[str] = Field(
        default=["json", "csv"],
        description=(
            "Export formats for the final summary:\n"
            "  ``json``      — structured summary with all metrics\n"
            "  ``csv``       — flat CSV suitable for spreadsheets\n"
            "  ``jsonl_raw`` — one JSON line per request (large; for custom analysis)"
        ),
    )
    markdown_report: bool = Field(
        default=True,
        description="Write a human-readable Markdown summary (``report.md``).",
    )
    save_responses: bool = Field(
        default=False,
        description=(
            "Persist the full generated text for every request. "
            "Can be large; useful for quality audits."
        ),
    )
    cluster: str = Field(
        default="",
        description="Cluster identifier stamped into artifact directory names (e.g. 'dgxa100-01').",
    )
    gpu: str = Field(
        default="",
        description="GPU type stamped into artifact directory names (e.g. 'H100-80GB').",
    )
    experiment_name: str = Field(
        default="",
        description="Experiment label stamped into artifact directory names for traceability.",
    )


class RunConfig(BaseModel):
    """
    Top-level configuration for a rag-perf benchmark run.

    There is a single CLI entrypoint::

        rag-perf -c configs/<preset>.yaml

    The YAML drives all behaviour. Each sweepable parameter is polymorphic —
    scalar for a single value, list for an axis — and the full grid is the
    Cartesian product of whichever fields are lists:

      - ``load.concurrency`` (concurrency axis)
      - ``rag.vdb_top_k``    (retrieval-fanout axis)
      - ``rag.reranker_top_k`` (reranker-fanout axis)

    Repeat-count and inter-point sleep live on ``load`` (``iterations``,
    ``sleep_between_points_s``). Toggle the load-test phase with
    ``aiperf.enabled``.

    Override specific fields programmatically (double-underscore = nested key)::

        config = RunConfig.from_yaml("configs/sweep.yaml")
        config = config.with_overrides(load__concurrency=16, rag__vdb_top_k=50)
    """

    target: TargetConfig = Field(default_factory=TargetConfig)
    aiperf: AiperfConfig = Field(default_factory=AiperfConfig)
    load: LoadConfig = Field(default_factory=LoadConfig)
    rag: RagParams = Field(default_factory=RagParams)
    generation: GenerationParams = Field(default_factory=GenerationParams)
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    model_name: str = Field(
        default="nvidia/nemotron-3-super-120b-a12b",
        description=(
            "Model name passed to aiperf via -m. Should match the model ID served "
            "by the RAG server's backend LLM (from APP_LLM_MODELNAME in the "
            "docker-compose). Used for request labelling; the RAG server ignores it."
        ),
    )
    tokenizer: str = Field(
        default="",
        description=(
            "HuggingFace tokenizer model ID passed to aiperf via --tokenizer. "
            "When empty, aiperf defaults to server-reported token counts. "
            "Set this when the HF tokenizer repo differs from the inference model name "
            "(e.g. 'meta-llama/Llama-3.3-70B-Instruct' for a NIM served as "
            "'nvidia/llama-3.3-nemotron-super-49b-v1.5')."
        ),
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load and validate configuration from a YAML file."""
        yaml = YAML(typ="safe")
        with open(path) as fh:
            data = yaml.load(fh)
        return cls.model_validate(data or {})

    @classmethod
    def defaults(cls) -> RunConfig:
        """Return a ``RunConfig`` populated with all default values."""
        return cls()

    def with_overrides(self, **kwargs: Any) -> RunConfig:
        """
        Return a *new* RunConfig with the given overrides applied.

        Uses double-underscore notation for nested keys::

            config.with_overrides(load__concurrency=16, rag__vdb_top_k=50)

        Raises ``KeyError`` if any override key does not correspond to an
        existing field path.
        """
        data = self.model_dump()
        for dotted_key, value in kwargs.items():
            parts = dotted_key.split("__")
            node = data
            for part in parts[:-1]:
                if part not in node:
                    raise KeyError(
                        f"Unknown config key segment '{part}' in '{dotted_key}'"
                    )
                node = node[part]
            leaf = parts[-1]
            if leaf not in node:
                raise KeyError(f"Unknown config key '{leaf}' in '{dotted_key}'")
            node[leaf] = value
        return RunConfig.model_validate(data)

    def to_yaml_str(self) -> str:
        """Serialise this config back to a YAML string."""
        buf = io.StringIO()
        yaml = YAML()
        yaml.dump(self.model_dump(mode="json"), buf)
        return buf.getvalue()

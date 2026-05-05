# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Agentic RAG configuration classes.

Imported by nvidia_rag.utils.configuration and placed under
NvidiaRAGConfig.agentic_rag.  Kept in a dedicated module to avoid
bloating configuration.py.

Import note
-----------
This module imports _ConfigBase and Field from nvidia_rag.utils.configuration.
That creates a deliberate partial-module circular reference: configuration.py
imports this file only after _ConfigBase and Field are fully defined, so
Python's partial-module resolution finds them correctly.  Do not move the
import in configuration.py above those definitions.
"""

from __future__ import annotations

from typing import Any, Union

from pydantic import SecretStr, field_validator
from pydantic import Field as PydanticField

# _ConfigBase and Field are imported from configuration.py.
# This works because configuration.py places its import of this module
# after _ConfigBase and Field are already defined.
from nvidia_rag.utils.configuration import _ConfigBase, Field


# =============================================================================
# AGENT BEHAVIOUR SUB-CONFIGS
# =============================================================================


class AgenticPlannerConfig(_ConfigBase):
    """Planner sub-config for agentic RAG."""

    max_plan_tasks: int = Field(
        default=5,
        env="AGENTIC_PLANNER_MAX_TASKS",
        description="Hard cap on answer tasks per plan.",
    )
    max_scope_rounds: int = Field(
        default=2,
        env="AGENTIC_PLANNER_MAX_SCOPE_ROUNDS",
        description="Scope discovery rounds before forcing answer phase.",
    )
    max_attempts: int = Field(
        default=3,
        env="AGENTIC_PLANNER_MAX_ATTEMPTS",
        description="Planner JSON parse/retry attempts.",
    )


class AgenticTaskExecutionConfig(_ConfigBase):
    """Task-execution sub-config for agentic RAG."""

    scope_max_retries: int = Field(
        default=1,
        env="AGENTIC_TASK_SCOPE_MAX_RETRIES",
        description="Retry budget for scope discovery tasks.",
    )
    answer_max_retries: int = Field(
        default=3,
        env="AGENTIC_TASK_ANSWER_MAX_RETRIES",
        description="Retry budget for answer tasks.",
    )


class AgenticLLMTransportConfig(_ConfigBase):
    """LLM transport sub-config for agentic RAG."""

    call_timeout: int = Field(
        default=300,
        env="AGENTIC_LLM_CALL_TIMEOUT",
        description="Per-call timeout in seconds.",
    )
    max_retries: int = Field(
        default=4,
        env="AGENTIC_LLM_MAX_RETRIES",
        description="Transport-level retries on timeout/5xx.",
    )


class AgenticVerificationConfig(_ConfigBase):
    """Verification sub-config for agentic RAG."""

    enabled: bool = Field(
        default=True,
        env="AGENTIC_VERIFICATION_ENABLED",
        description="Toggle verification pass on/off.",
    )
    max_tasks: int = Field(
        default=3,
        env="AGENTIC_VERIFICATION_MAX_TASKS",
        description="Max verification follow-up tasks.",
    )


class AgenticContextConfig(_ConfigBase):
    """Context sub-config for agentic RAG."""

    max_tokens: int = Field(
        default=100000,
        env="AGENTIC_CONTEXT_MAX_TOKENS",
        description="Token budget for chunk context in prompts.",
    )


# =============================================================================
# PER-ROLE LLM CONFIGS
# Each role gets its own env var prefix so every LLM can be configured
# independently.  If a role's model_name is left empty the builder falls
# back to the planner LLM config.
#
# Env var naming convention:
#   AGENTIC_{ROLE}_LLM_SERVERURL
#   AGENTIC_{ROLE}_LLM_MODEL
#   AGENTIC_{ROLE}_LLM_APIKEY
# =============================================================================


class AgenticPlannerLLMConfig(_ConfigBase):
    """LLM config for the planner role (scope resolution + task creation).

    Env vars: AGENTIC_PLANNER_LLM_SERVERURL, AGENTIC_PLANNER_LLM_MODEL,
              AGENTIC_PLANNER_LLM_APIKEY
    """

    server_url: str = Field(
        default="",
        env="AGENTIC_PLANNER_LLM_SERVERURL",
        description="URL endpoint for the planner LLM service.",
    )
    model_name: str = Field(
        default="",
        env="AGENTIC_PLANNER_LLM_MODEL",
        description="Model name for the planner LLM. Falls back to main LLM if empty.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="AGENTIC_PLANNER_LLM_APIKEY",
        description="API key for the planner LLM (overrides global NVIDIA_API_KEY).",
    )

    @field_validator("server_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v


class AgenticTaskLLMConfig(_ConfigBase):
    """LLM config for the task role (answering individual sub-questions).

    Env vars: AGENTIC_TASK_LLM_SERVERURL, AGENTIC_TASK_LLM_MODEL,
              AGENTIC_TASK_LLM_APIKEY
    """

    server_url: str = Field(
        default="",
        env="AGENTIC_TASK_LLM_SERVERURL",
        description="URL endpoint for the task LLM service.",
    )
    model_name: str = Field(
        default="",
        env="AGENTIC_TASK_LLM_MODEL",
        description="Model name for the task LLM. Falls back to planner LLM if empty.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="AGENTIC_TASK_LLM_APIKEY",
        description="API key for the task LLM (overrides global NVIDIA_API_KEY).",
    )

    @field_validator("server_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v


class AgenticSeedGenLLMConfig(_ConfigBase):
    """LLM config for the seed-gen role (retry seed query generation).

    Env vars: AGENTIC_SEED_GEN_LLM_SERVERURL, AGENTIC_SEED_GEN_LLM_MODEL,
              AGENTIC_SEED_GEN_LLM_APIKEY
    """

    server_url: str = Field(
        default="",
        env="AGENTIC_SEED_GEN_LLM_SERVERURL",
        description="URL endpoint for the seed-gen LLM service.",
    )
    model_name: str = Field(
        default="",
        env="AGENTIC_SEED_GEN_LLM_MODEL",
        description="Model name for the seed-gen LLM. Falls back to planner LLM if empty.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="AGENTIC_SEED_GEN_LLM_APIKEY",
        description="API key for the seed-gen LLM (overrides global NVIDIA_API_KEY).",
    )

    @field_validator("server_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v


class AgenticSynthesisLLMConfig(_ConfigBase):
    """LLM config for the synthesis role (final answer generation).

    Env vars: AGENTIC_SYNTHESIS_LLM_SERVERURL, AGENTIC_SYNTHESIS_LLM_MODEL,
              AGENTIC_SYNTHESIS_LLM_APIKEY
    """

    server_url: str = Field(
        default="",
        env="AGENTIC_SYNTHESIS_LLM_SERVERURL",
        description="URL endpoint for the synthesis LLM service.",
    )
    model_name: str = Field(
        default="",
        env="AGENTIC_SYNTHESIS_LLM_MODEL",
        description="Model name for the synthesis LLM. Falls back to planner LLM if empty.",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="AGENTIC_SYNTHESIS_LLM_APIKEY",
        description="API key for the synthesis LLM (overrides global NVIDIA_API_KEY).",
    )

    @field_validator("server_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v


# Union type used for type hints in builder.py.
AgenticAnyLLMConfig = Union[
    AgenticPlannerLLMConfig,
    AgenticTaskLLMConfig,
    AgenticSeedGenLLMConfig,
    AgenticSynthesisLLMConfig,
]


# =============================================================================
# TOP-LEVEL AGENTIC RAG CONFIG
# =============================================================================


class AgenticRAGConfig(_ConfigBase):
    """Configuration for the agentic RAG flow (NvidiaRAG(agentic=True)).

    Each LLM role has its own config class with its own env var prefix.
    If a role's model_name is left empty the builder falls back to the
    planner_llm config, so a minimal deployment only needs to set
    AGENTIC_PLANNER_LLM_* variables.

    Environment variable prefixes:
        Planner LLM   — AGENTIC_PLANNER_LLM_*
        Task LLM      — AGENTIC_TASK_LLM_*
        Seed-gen LLM  — AGENTIC_SEED_GEN_LLM_*
        Synthesis LLM — AGENTIC_SYNTHESIS_LLM_*
        Agent tuning  — AGENTIC_PLANNER_*, AGENTIC_TASK_*, AGENTIC_LLM_*,
                        AGENTIC_VERIFICATION_*, AGENTIC_CONTEXT_*
        Behaviour     — AGENTIC_CONCURRENCY_LIMIT, AGENTIC_RECURSION_LIMIT,
                        AGENTIC_LOG_LEVEL
        Tracing       — AGENTIC_JSON_TRACES, AGENTIC_TRACES_DIR
    """

    # --- Per-role LLM configs -----------------------------------------------
    planner_llm: AgenticPlannerLLMConfig = PydanticField(
        default_factory=AgenticPlannerLLMConfig,
        description="LLM for planning (scope resolution + task creation).",
    )
    task_llm: AgenticTaskLLMConfig = PydanticField(
        default_factory=AgenticTaskLLMConfig,
        description="LLM for answering sub-questions. Falls back to planner_llm if model_name is empty.",
    )
    seed_gen_llm: AgenticSeedGenLLMConfig = PydanticField(
        default_factory=AgenticSeedGenLLMConfig,
        description="LLM for generating retry seed queries. Falls back to planner_llm if model_name is empty.",
    )
    synthesis_llm: AgenticSynthesisLLMConfig = PydanticField(
        default_factory=AgenticSynthesisLLMConfig,
        description="LLM for final answer synthesis. Falls back to planner_llm if model_name is empty.",
    )

    # --- Agent behaviour ----------------------------------------------------
    concurrency_limit: int = Field(
        default=3,
        env="AGENTIC_CONCURRENCY_LIMIT",
        description="Max concurrent LLM/retrieval calls inside the agent.",
    )
    recursion_limit: int = Field(
        default=50,
        env="AGENTIC_RECURSION_LIMIT",
        description="Max recursion depth for the LangGraph workflow.",
    )
    log_level: str = Field(
        default="INFO",
        env="AGENTIC_LOG_LEVEL",
        description="Logging level for the agentic RAG agent (DEBUG/INFO/WARNING/ERROR).",
    )

    # --- JSON trace output --------------------------------------------------
    json_traces: bool = Field(
        default=False,
        env="AGENTIC_JSON_TRACES",
        description="Write per-query JSON trace files to json_traces_output_dir.",
    )
    json_traces_output_dir: str = Field(
        default="./traces",
        env="AGENTIC_TRACES_DIR",
        description="Directory for per-query JSON trace files.",
    )

    # --- Component sub-configs ----------------------------------------------
    planner: AgenticPlannerConfig = PydanticField(
        default_factory=AgenticPlannerConfig
    )
    task_execution: AgenticTaskExecutionConfig = PydanticField(
        default_factory=AgenticTaskExecutionConfig
    )
    llm: AgenticLLMTransportConfig = PydanticField(
        default_factory=AgenticLLMTransportConfig
    )
    verification: AgenticVerificationConfig = PydanticField(
        default_factory=AgenticVerificationConfig
    )
    context: AgenticContextConfig = PydanticField(
        default_factory=AgenticContextConfig
    )

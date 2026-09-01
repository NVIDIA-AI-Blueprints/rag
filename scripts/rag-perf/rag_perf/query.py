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
Query input loading and synthetic generation.

Two service classes:

- ``SyntheticQueryGenerator`` — calls an OpenAI-compatible chat completions
  endpoint to produce ``cfg.num_queries`` synthetic query strings, using
  either ``random`` or ``dataset_based`` strategies.

- ``QueryLoader`` — single entry point used by the orchestrator. Dispatches
  to the right loader based on ``input.source`` (jsonl / csv / synthetic),
  samples or cycles to ``count`` entries, then merges per-query overrides
  into run-level ``RagParams`` / ``GenerationParams`` defaults to produce
  request dicts ready for ``POST /v1/generate`` and ``aiperf``.
"""

from __future__ import annotations

import asyncio
import csv
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from ruamel.yaml import YAML

from rag_perf.config import (
    GenerationParams,
    InputConfig,
    InputSource,
    RagParams,
    SyntheticInputConfig,
    SyntheticMode,
)

_console = Console()

# Prompt YAML resolution order: explicit config path → CWD → packaged default.
_DEFAULT_PROMPTS_CWD = Path("prompts/default_prompts.yaml")
_DEFAULT_PROMPTS_PKG = Path(__file__).parent.parent / "prompts" / "default_prompts.yaml"

# Per-query override field sets — derived from the run-level config models.
# Used by ``QueryLoader._build_request`` to merge per-query overrides into defaults.
_RAG_OVERRIDE_FIELDS: set[str] = set(RagParams.model_fields.keys())
_GEN_OVERRIDE_FIELDS: set[str] = set(GenerationParams.model_fields.keys())
_ALL_OVERRIDE_FIELDS: set[str] = _RAG_OVERRIDE_FIELDS | _GEN_OVERRIDE_FIELDS


@dataclass
class PromptTemplates:
    """
    Prompt templates for one generation mode (``random`` or ``dataset_based``).

    Templates use Python ``str.format`` with named substitution variables:
    - ``{word_target}``  — approximate minimum word count per query
    - ``{index}``        — 1-based query sequence number
    - ``{ref}``          — reference question (``dataset_based`` mode only)
    """

    system: str
    user: str


class SyntheticQueryGenerator:
    """
    LLM-based synthetic query generation.

    Two strategies: ``random`` (no reference material) and ``dataset_based``
    (generates queries inspired by reference questions from a dataset file).
    """

    @staticmethod
    def generate(
        cfg: SyntheticInputConfig,
        output_path: Path | None = None,
    ) -> list[str]:
        """
        Generate ``cfg.num_queries`` synthetic query strings by calling the LLM.

        LLM calls fan out concurrently up to ``cfg.generation_concurrency``
        in-flight requests. When ``output_path`` is given, each successfully
        generated query is appended to the JSONL **as it completes** (with
        flush) — so a mid-generation failure preserves everything that was
        finished, instead of losing the whole batch.

        Args:
            cfg:         Synthetic input configuration (mode, LLM URL, prompts,
                         concurrency, etc.).
            output_path: Optional JSONL path to stream into. None → in-memory
                         only.

        Returns:
            List of generated query strings, in completion order (which may
            differ from request order when concurrency > 1).
        """
        model = SyntheticQueryGenerator._resolve_model(cfg)
        prompts = SyntheticQueryGenerator._load_prompts(
            cfg.mode.value, cfg.prompts_file
        )
        word_target = SyntheticQueryGenerator._word_target(cfg.min_query_tokens)
        system_msg = prompts.system.format(word_target=word_target)

        if cfg.mode == SyntheticMode.RANDOM:
            per_query_user = [
                prompts.user.format(word_target=word_target, index=i + 1)
                for i in range(cfg.num_queries)
            ]
            mode_label = "Random"
        else:
            dataset_path = SyntheticQueryGenerator._resolve_dataset_file(cfg)
            _console.print(
                f"  [dim]Loading reference questions from {dataset_path}[/dim]"
            )
            with open(dataset_path) as fh:
                data = json.load(fh)
            ref_questions = SyntheticQueryGenerator._extract_questions(
                data, dataset_path
            )
            if not ref_questions:
                raise ValueError(f"No questions found in {dataset_path}")
            rng = random.Random(42)
            per_query_user = [
                prompts.user.format(
                    word_target=word_target,
                    index=i + 1,
                    ref=rng.choice(ref_questions),
                )
                for i in range(cfg.num_queries)
            ]
            mode_label = "Dataset-based"

        _console.print(
            f"  [cyan]→[/cyan]  Generating [bold]{cfg.num_queries}[/bold] queries via "
            f"[dim]{cfg.llm_url}[/dim] (model: [dim]{model}[/dim], "
            f"concurrency: [dim]{cfg.generation_concurrency}[/dim]) ..."
        )

        return asyncio.run(
            SyntheticQueryGenerator._generate_async(
                cfg=cfg,
                model=model,
                system_msg=system_msg,
                per_query_user=per_query_user,
                output_path=output_path,
                mode_label=mode_label,
            )
        )

    @staticmethod
    async def _generate_async(
        cfg: SyntheticInputConfig,
        model: str,
        system_msg: str,
        per_query_user: list[str],
        output_path: Path | None,
        mode_label: str,
    ) -> list[str]:
        """Concurrent generation orchestrator with streaming write.

        Each task runs the sync ``_call_llm`` in a worker thread (so the
        existing ``httpx.post`` call site doesn't change), bounded by an
        asyncio Semaphore. Successful queries are appended to ``output_path``
        immediately under a write lock, then accumulated in the returned list.
        On the first task failure, ``asyncio.gather`` cancels the rest and
        the file is closed in a ``finally`` — partial progress is preserved.
        """
        sem = asyncio.Semaphore(cfg.generation_concurrency)
        write_lock = asyncio.Lock()
        queries: list[str] = []
        n = len(per_query_user)

        # Build the extra-body payload once: disable_thinking sets
        # chat_template_kwargs.enable_thinking=False (suppressing reasoning on
        # Nemotron Omni / Qwen-Reasoning), and any user-supplied extra_body
        # is merged on top so explicit keys win.
        extra_body: dict[str, Any] = {}
        if cfg.disable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if cfg.extra_body:
            extra_body.update(cfg.extra_body)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fh = output_path.open("w")
        else:
            fh = None

        async def one(i: int, user_msg: str) -> None:
            async with sem:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
                try:
                    raw = await asyncio.to_thread(
                        SyntheticQueryGenerator._call_llm,
                        cfg.llm_url,
                        model,
                        messages,
                        cfg.min_query_tokens,
                        cfg.temperature,
                        extra_body or None,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"{mode_label} synthetic query generation failed at "
                        f"query {i + 1}: {exc}"
                    ) from exc
                # Strip any leading numbering chars ("1.", "2)", etc.) — multi-char lstrip is intentional.
                q = raw.lstrip("0123456789.). ").strip()  # noqa: B005
                async with write_lock:
                    queries.append(q)
                    if fh is not None:
                        fh.write(json.dumps({"query": q}) + "\n")
                        fh.flush()
                    _console.print(
                        f"  [dim]  Generated {len(queries)}/{n}[/dim]", end="\r"
                    )

        try:
            await asyncio.gather(*(one(i, msg) for i, msg in enumerate(per_query_user)))
        finally:
            if fh is not None:
                fh.close()

        _console.print()
        return queries

    @staticmethod
    def _load_prompts(mode: str, prompts_file: str | None) -> PromptTemplates:
        """Load prompt templates for ``mode`` from the YAML file."""
        path: Path | None = None
        if prompts_file:
            path = Path(prompts_file)
            if not path.exists():
                raise FileNotFoundError(
                    f"Prompts file not found: {prompts_file}\n"
                    f"Update synthetic.prompts_file in your config YAML."
                )
        elif _DEFAULT_PROMPTS_CWD.exists():
            path = _DEFAULT_PROMPTS_CWD
        elif _DEFAULT_PROMPTS_PKG.exists():
            path = _DEFAULT_PROMPTS_PKG
        else:
            raise FileNotFoundError(
                "No prompts YAML found.  Either:\n"
                "  1. Set synthetic.prompts_file in your config YAML, or\n"
                f"  2. Create {_DEFAULT_PROMPTS_CWD} in the current directory."
            )

        yaml = YAML(typ="safe")
        with open(path) as fh:
            data = yaml.load(fh)

        if mode not in data:
            raise KeyError(
                f"Mode '{mode}' not found in prompts file {path}. "
                f"Expected keys: {list(data.keys())}"
            )

        section = data[mode]
        return PromptTemplates(system=section["system"], user=section["user"])

    @staticmethod
    def _resolve_dataset_file(cfg: SyntheticInputConfig) -> str:
        """Resolve the path to the dataset file for ``dataset_based`` mode."""
        if cfg.dataset_file:
            if not Path(cfg.dataset_file).exists():
                raise FileNotFoundError(
                    f"Dataset file not found: {cfg.dataset_file}\n"
                    f"Update synthetic.dataset_file in your config YAML."
                )
            return cfg.dataset_file

        if cfg.dataset_name:
            datasets_dir = Path("./datasets")
            candidates = [
                datasets_dir / cfg.dataset_name / "train.json",
                datasets_dir / f"{cfg.dataset_name}.json",
                datasets_dir / cfg.dataset_name / "data.json",
            ]
            for candidate in candidates:
                if candidate.exists():
                    _console.print(f"  [dim]Auto-found dataset: {candidate}[/dim]")
                    return str(candidate)

            raise FileNotFoundError(
                f"Dataset '{cfg.dataset_name}' not found.  Searched:\n"
                + "\n".join(f"  {c}" for c in candidates)
                + "\n\nPlace your dataset file in the ./datasets/ directory or set "
                "synthetic.dataset_file to an explicit path."
            )

        raise ValueError(
            "For dataset_based mode, set synthetic.dataset_file (explicit path) "
            "or synthetic.dataset_name (auto-lookup under ./datasets/) in your config YAML."
        )

    @staticmethod
    def _resolve_model(cfg: SyntheticInputConfig) -> str:
        """Return the model name, auto-discovering it from /v1/models if not set."""
        if cfg.llm_model:
            return cfg.llm_model

        base = cfg.llm_url.replace("/chat/completions", "").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        try:
            resp = httpx.get(f"{base}/v1/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0]["id"]
        except Exception:
            pass
        return "local"

    @staticmethod
    def _call_llm(
        url: str,
        model: str,
        messages: list[dict[str, Any]],
        token_target: int = 512,
        temperature: float = 0.9,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Make a single chat completions call and return the stripped content.

        Reads only ``message.content`` — never falls back to ``reasoning_content``,
        which on reasoning models contains the chain-of-thought, not the answer.
        If ``content`` is empty (typically: the model exhausted ``max_tokens`` on
        reasoning), the call fails so the orchestrator can surface it instead of
        polluting the JSONL with deliberation text.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": token_target,
            "min_tokens": token_target,
        }
        if extra_body:
            payload.update(extra_body)
        resp = httpx.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        if not content:
            reasoning_present = bool((msg.get("reasoning_content") or "").strip())
            raise RuntimeError(
                "LLM returned empty content"
                + (
                    " (reasoning_content was populated — model exhausted its "
                    "budget on chain-of-thought; raise min_query_tokens or set "
                    "synthetic.disable_thinking=true)."
                    if reasoning_present
                    else f". message fields: {list(msg.keys())}."
                )
            )
        return content

    @staticmethod
    def _word_target(min_tokens: int) -> int:
        """Convert a token target to an approximate word count (1 token ≈ 0.75 words)."""
        return max(10, int(min_tokens * 0.75))

    @staticmethod
    def _extract_questions(data: object, path: str) -> list[str]:
        """Extract question strings from a dataset file."""
        items: list[dict[str, Any]] = []
        if isinstance(data, list):
            items = [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            for key in ("data", "questions", "examples"):
                if key in data and isinstance(data[key], list):
                    items = [x for x in data[key] if isinstance(x, dict)]
                    break
            if not items:
                items = [v for v in data.values() if isinstance(v, dict)]

        questions: list[str] = []
        for item in items:
            for key in ("question", "query", "input", "text"):
                if key in item and isinstance(item[key], str) and item[key].strip():
                    questions.append(item[key].strip())
                    break
        return questions


class QueryLoader:
    """
    Query input loading and format conversion.

    Handles all three input modes (JSONL, CSV, synthetic) and produces
    fully-resolved request dicts ready for the profiler or aiperf.
    """

    @staticmethod
    def load(
        cfg: InputConfig,
        rag_params: RagParams,
        gen_params: GenerationParams,
        count: int,
    ) -> list[dict[str, Any]]:
        """
        Load and expand queries into a list of fully-resolved request dicts.

        Args:
            cfg:        Input configuration (source, file path, sampling, etc.)
            rag_params: Run-level RAG parameter defaults.
            gen_params: Run-level generation parameter defaults.
            count:      Total number of request dicts to return.

        Returns:
            List of ``count`` request dicts ready for the profiler or aiperf.
        """
        raw = QueryLoader._load_raw_queries(cfg)
        sampled = QueryLoader._sample(raw, count, cfg.sampling, cfg.seed)
        return [QueryLoader._build_request(q, rag_params, gen_params) for q in sampled]

    @staticmethod
    def write_sharegpt(queries: list[dict[str, Any]], path: str | Path) -> Path:
        """
        Write queries to a single-turn JSONL file that aiperf can consume.

        aiperf's ``single-turn`` dataset loader expects one JSON object per line::

            {"text": "<query text>"}
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for q in queries:
                user_text = next(
                    (
                        m["content"]
                        for m in q.get("messages", [])
                        if m["role"] == "user"
                    ),
                    "",
                )
                fh.write(json.dumps({"text": user_text}) + "\n")
        return out

    @staticmethod
    def make_temp(queries: list[dict[str, Any]]) -> str:
        """
        Write queries to a NamedTemporaryFile and return its path.

        The file is *not* deleted automatically — the caller should clean it up
        after aiperf finishes (or let the OS handle it on process exit).
        """
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", prefix="rag_perf_queries_", delete=False
        )
        tmp.close()
        QueryLoader.write_sharegpt(queries, tmp.name)
        return tmp.name

    @staticmethod
    def _load_raw_queries(cfg: InputConfig) -> list[dict[str, Any]]:
        """Return raw query dicts based on the configured source."""
        if cfg.source == InputSource.JSONL:
            return QueryLoader._load_jsonl(cfg.file)  # type: ignore[arg-type]
        elif cfg.source == InputSource.CSV:
            return QueryLoader._load_csv(cfg.file)  # type: ignore[arg-type]
        elif cfg.source == InputSource.SYNTHETIC:
            return QueryLoader._generate_and_dump(cfg)
        else:
            raise ValueError(f"Unknown input source: {cfg.source!r}")

    @staticmethod
    def _generate_and_dump(cfg: InputConfig) -> list[dict[str, Any]]:
        """Generate synthetic queries, write them to a JSONL file, then load from it."""
        syn = cfg.synthetic
        _console.print(
            f"  [cyan]→[/cyan]  Generating [bold]{syn.num_queries}[/bold] synthetic queries "
            f"(mode: [bold]{syn.mode.value}[/bold]) ..."
        )

        out_path = Path(syn.jsonl_output_path)
        # generate() streams each completed query to disk under a write lock,
        # so a mid-generation failure leaves the file with all successful
        # queries up to that point.
        queries = SyntheticQueryGenerator.generate(syn, output_path=out_path)

        _console.print(
            f"  [green]✔[/green]  {len(queries)} queries written to "
            f"[cyan]{out_path}[/cyan]"
        )

        return [{"query": q} for q in queries]

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        """
        Load queries from a .jsonl file.

        Each non-blank, non-comment line must be a JSON object containing at
        least a ``query`` key.

        Raises:
            ValueError: If the file is empty, a line has invalid JSON, or a line
                        is missing the required ``query`` key.
        """
        queries = []
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno} — invalid JSON: {exc}") from exc
                if "query" not in obj:
                    raise ValueError(f"{path}:{lineno} — missing required 'query' key")
                queries.append(obj)
        if not queries:
            raise ValueError(f"No queries found in {path}")
        return queries

    @staticmethod
    def _load_csv(path: str) -> list[dict[str, Any]]:
        """
        Load queries from a CSV file.

        The file must contain at least a ``query`` column.  Any other column
        whose name matches a ``RagParams`` or ``GenerationParams`` field is
        treated as a per-query override.

        Raises:
            ValueError: If the ``query`` column is absent.
        """
        queries = []
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "query" not in reader.fieldnames:
                raise ValueError(f"{path} — CSV must have a 'query' column")
            for row in reader:
                obj: dict[str, Any] = {"query": row["query"]}
                for f in _ALL_OVERRIDE_FIELDS:
                    if f in row and row[f]:
                        obj[f] = QueryLoader._try_parse(row[f])
                queries.append(obj)
        return queries

    @staticmethod
    def _sample(
        queries: list[dict[str, Any]],
        count: int,
        strategy: str,
        seed: int,
    ) -> list[dict[str, Any]]:
        """
        Expand or subsample ``queries`` to exactly ``count`` entries.

        Args:
            queries:  Raw query dicts to sample from.
            count:    Target number of entries in the result.
            strategy: ``"random"`` | ``"sequential"`` | ``"shuffle-once"``
            seed:     Random seed for reproducibility.
        """
        if strategy == "shuffle-once":
            rng = random.Random(seed)
            shuffled = list(queries)
            rng.shuffle(shuffled)
            source = shuffled
        elif strategy == "sequential":
            source = list(queries)
        else:  # random (default)
            rng = random.Random(seed)
            return [rng.choice(queries) for _ in range(count)]

        result = []
        idx = 0
        while len(result) < count:
            result.append(source[idx % len(source)])
            idx += 1
        return result

    @staticmethod
    def _build_request(
        raw: dict[str, Any],
        rag_defaults: RagParams,
        gen_defaults: GenerationParams,
    ) -> dict[str, Any]:
        """
        Merge per-query overrides into run-level defaults and build the full
        request dict that will be sent to ``POST /v1/generate``.
        """
        rag_data = rag_defaults.model_dump()
        gen_data = gen_defaults.model_dump()

        for f in _RAG_OVERRIDE_FIELDS:
            if f in raw:
                rag_data[f] = raw[f]
        for f in _GEN_OVERRIDE_FIELDS:
            if f in raw:
                gen_data[f] = raw[f]

        # Drop None-valued fields so the server falls back to its own defaults.
        # Some Prompt fields (e.g. min_tokens) are typed `int` (not `int | None`)
        # and FastAPI rejects an explicit null with 422.
        rag_data = {k: v for k, v in rag_data.items() if v is not None}
        gen_data = {k: v for k, v in gen_data.items() if v is not None}

        messages: list[dict[str, str]] = [{"role": "user", "content": raw["query"]}]

        return {
            "messages": messages,
            "stream": True,
            **rag_data,
            **gen_data,
        }

    @staticmethod
    def _try_parse(value: str) -> Any:
        """Attempt to parse a CSV string cell as JSON; fall back to raw string."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value

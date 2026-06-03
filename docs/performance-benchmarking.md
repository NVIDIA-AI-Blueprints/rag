<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Benchmark the Performance of Your NVIDIA RAG Blueprint System

After you [deploy your NVIDIA RAG Blueprint system](readme.md#deployment-options-for-rag-blueprint),
benchmark its performance: latency, throughput, and per-stage timing: using the bundled `rag-perf` CLI.

For accuracy benchmarks (RAGAS-based scoring of answer quality), see [`scripts/eval/README.md`](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/scripts/eval/README.md): the runnable `evaluate_rag.py` CLI. (For the conceptual / notebook overview of RAGAS metrics, see [Evaluate Your NVIDIA RAG Blueprint System](evaluate.md).) The two tools are complementary: `evaluate_rag.py` measures *how well* the system answers; `rag-perf` measures *how fast and at what concurrency*.

## What `rag-perf` measures

For each benchmark point, `rag-perf` runs two passes against the deployed RAG server and folds the results into a unified report:

1. **Profiling pass**: direct async httpx requests against the RAG server. Captures **server-side per-stage timing** that a generic load tool cannot see: time spent in retrieval, reranking, and LLM TTFT, plus citation counts and relevance scores from the streamed response. From this pass `rag-perf` infers which stage is the **bottleneck** for the current configuration (`retrieval`, `reranking`, or `llm`).
2. **Load-test pass**: [aiperf](https://github.com/NVIDIA/aiperf) drives concurrent traffic against the same server through the bundled `nvidia_rag` endpoint plugin. Captures TTFT mean / p50 / p90 / p99, end-to-end latency p90 / p99, output-token throughput, request throughput, and error rate.

The combined output is a single `RagMetricsSummary` rendered as a Rich terminal table, a Markdown report, and structured JSON / CSV for downstream graphing. Set `aiperf.enabled: false` in the YAML to skip the load-test pass entirely: useful for fast iteration on retrieval/reranker tuning.


## Quickstart

This section runs a full benchmark in under three minutes against a default deployment using the queries shipped with the tool (`scripts/rag-perf/examples/queries.jsonl`).

**Prerequisites**: the RAG server must be running and reachable on the network: for example, after completing the [Quickstart: self-hosted Docker](deploy-docker-self-hosted.md). Python ≥ 3.11 on the machine running the benchmark.

```bash
# 1. Install rag-perf into its own uv-managed venv (one-time).
uv sync --project scripts/rag-perf

# 2. Edit configs/single_run.yaml to point at your collection (rag.collection_names),
#    then run it.
uv run --project scripts/rag-perf rag-perf -c scripts/rag-perf/configs/single_run.yaml

# 3. View the report.
ls rag-perf-results/single_run/run_*/
# report.md  results.csv  results.json  profiling/  aiperf_rag_on/
```

You should see a Rich-rendered table on stdout with the bottleneck stage, TTFT percentiles, throughput, and error rate. The `report.md` file contains the same data in Markdown form for sharing or PR attachments.

> **Tip:** copy the preset to your own YAML (for example `cp configs/single_run.yaml my_run.yaml`) and edit fields there. The CLI takes only `--config`, so the YAML is the unit of versionable configuration.


## Three preset workflows

`rag-perf` is a single command: `rag-perf -c <config>`: and behaviour is fully driven by the YAML you pass it. Three presets cover the common workflows; each section below describes when to use it, what it produces, and how to invoke it.

| Preset | When to use | Approximate runtime |
|---|---|---|
| `quick_profile.yaml` | Iterating on retrieval / reranker tuning. No load test. | ~30 s |
| `single_run.yaml` | One concurrency level; full report (profiling + load test). | ~2 min |
| `sweep.yaml` | Compare across multiple values of any axis. Make `load.concurrency`, `rag.vdb_top_k`, or `rag.reranker_top_k` a list to sweep that axis; multiple lists give a full Cartesian sweep. | A few minutes per point. |


### Quick profiling

Use this when you want server-side stage timing fast: for example to check whether retrieval or reranking is the bottleneck after changing `vdb_top_k`. No load is generated.

Config: [`scripts/rag-perf/configs/quick_profile.yaml`](../scripts/rag-perf/configs/quick_profile.yaml).

> **Before you run:** edit `rag.collection_names` in the config to point at a real collection on your deployed ingestor server. The shipped value is `["<collection_name>"]`, which the run will fail to retrieve from.

```bash
uv run --project scripts/rag-perf rag-perf -c scripts/rag-perf/configs/quick_profile.yaml
```

Output (under `output.dir`, default `./rag-perf-results/quick_profile/`):

```
run_<ts>/
├── profile_report.md
├── profile_results.json
└── profiling/
    └── profiler_records.jsonl
```

The `profile_*` filename prefix flags this as a profile-only run (no aiperf data). To convert any other config to profile-only, set `aiperf.enabled: false` in the YAML.


### Single-point run

Use this when you want a single benchmark point with both passes: for example a regression check at a known-good concurrency before launching a larger sweep.

Config: [`scripts/rag-perf/configs/single_run.yaml`](../scripts/rag-perf/configs/single_run.yaml).

> **Before you run:** edit `rag.collection_names` in the config to point at a real collection on your deployed ingestor server.

```bash
uv run --project scripts/rag-perf rag-perf -c scripts/rag-perf/configs/single_run.yaml
```

Output:

```
run_<ts>/
├── report.md
├── results.csv
├── results.json
├── profiling/
│   └── profiler_records.jsonl
└── aiperf_rag_on/
    ├── profile_export_aiperf.csv
    ├── profile_export_aiperf.json
    └── profile_export.jsonl
```


### Concurrency sweep

Use this to compare TTFT, latency, and throughput across multiple concurrency levels. The config's `load.concurrency` is a list (for example `[1, 4, 8, 16, 32]`); each value is a benchmark point. Edit the list in the YAML to add or remove levels.

Config: [`scripts/rag-perf/configs/sweep.yaml`](../scripts/rag-perf/configs/sweep.yaml).

> **Before you run:** edit `rag.collection_names` in the config to point at a real collection on your deployed ingestor server.

```bash
uv run --project scripts/rag-perf rag-perf -c scripts/rag-perf/configs/sweep.yaml
```

Output is **nested**: each grid point gets its own subdirectory, with aggregate report files at the run root:

```
run_<ts>/
├── report.md
├── results.csv             # one row per point
├── results.json
└── iter_1/
    ├── CR:1_ISL:50_OSL:512_VDB-K:20_RERANKER-K:4_Model:.../
    │   ├── profiling/
    │   └── aiperf_rag_on/
    ├── CR:4_ISL:50_OSL:512_VDB-K:20_RERANKER-K:4_Model:.../
    └── ...
```

When `load.iterations > 1`, the entire grid is repeated and each repetition writes to its own `iter_<i>/` subdirectory so variance can be measured across runs.

To run a full Cartesian sweep across concurrency × `vdb_top_k` × `reranker_top_k`, change any of those fields from a scalar to a list (for example `rag.vdb_top_k: [20, 100]`). For overnight runs, set `load.sleep_between_points_s: 60` so the server has time to drain in-flight requests between points (this matches the blueprint pipeline's default).


## What you will see on stdout

Every invocation prints, in order:

1. The **startup banner** plus a one-line summary of target / collection / top-k / input source / concurrency / aiperf state.
2. The **fully resolved configuration** as YAML, dumped verbatim so the run is reproducible from the terminal output alone (every field that drove the run, including the `synthetic` block and any defaults that were filled in).
3. For each grid point, a **per-iteration log line** plus the **aiperf shell command** in copy-pastable form (`$ python -m aiperf profile -m … --endpoint-type nvidia_rag …`) before the subprocess fires.
4. After each grid point completes, a **rich per-point summary table** with the full stage breakdown (retrieval / reranking / LLM TTFT) with percent-of-TTFT bars, citation count and relevance score, the inferred bottleneck, plus the load-test block (TTFT / E2E / throughput / error rate).
5. After all points finish, a **side-by-side comparison table** auto-labelled by whichever axis varied (concurrency / vdb_top_k / reranker_top_k / iter), and a one-liner identifying the optimal-throughput point.

The same data is also persisted under `output.dir/run_<ts>/`: see [Output layout](#output-layout): so terminal scrollback is not load-bearing.


## Query inputs

`rag-perf` needs a stream of queries to drive at the RAG server. The `input` block in the YAML chooses where they come from. **Set exactly one** of `input.file` or `input.synthetic`: they are mutually exclusive and validation fails if both are present. When neither is set, `synthetic` is auto-filled with defaults so a bare config still validates.

### File-based queries

Set `input.file` to a path. The format is auto-detected from the extension:

- **`.jsonl`**: one JSON object per line. Each object must have a `query` key. Any field also defined under `rag.*` or `generation.*` becomes a per-query override (so a single file can mix multiple collections, top-K values, max-token caps, and so on):
  ```json
  {"query": "What was NVIDIA revenue?", "collection_names": ["finance"]}
  {"query": "Summarize the 10-K risks.", "vdb_top_k": 50}
  {"query": "Show me chart-heavy pages.", "max_tokens": 1024}
  ```
- **`.csv`**: must have a `query` column. Other columns whose names match `rag.*` or `generation.*` field names become per-query overrides; CSV cell values are JSON-parsed when possible (so a cell like `["finance"]` is interpreted as a list, not a string).

If the requested number of requests exceeds the file's row count, `rag-perf` re-uses queries according to `input.sampling`:

- `random` (default): random with replacement.
- `sequential`: cycle through in order.
- `shuffle-once`: shuffle once, then cycle.

`input.seed` (default `42`) makes sampling reproducible across runs.

A small example file ships at [`scripts/rag-perf/examples/queries.jsonl`](../scripts/rag-perf/examples/queries.jsonl); the three preset configs all point at it by default.

### Synthetic queries (LLM-generated)

When `input.synthetic` is set, `rag-perf` calls an OpenAI-compatible chat-completions endpoint *before* the benchmark to generate `synthetic.num_queries` queries, writes them to `synthetic.jsonl_output_path`, and then runs the benchmark from that JSONL: so the run is reproducible even though the queries were generated.

Two modes:

- **`random`**: the LLM generates queries from scratch using the prompt templates in `synthetic.prompts_file` (or the bundled defaults at `scripts/rag-perf/prompts/default_prompts.yaml` if unset). Useful when you do not have a query corpus yet but want plausible questions to drive load against your collection.
- **`dataset_based`**: the LLM is seeded with reference questions from a JSON dataset and asked to produce variations. Set either `synthetic.dataset_file` (explicit path) or `synthetic.dataset_name` (auto-lookup under `./datasets/<name>/train.json` and so on). Validation fails if neither is set in `dataset_based` mode.

Key knobs:

| Field | Purpose |
|---|---|
| `synthetic.num_queries` | How many distinct queries to generate. The query list is cycled if `total_requests` exceeds it. |
| `synthetic.min_query_tokens` | Approximate minimum token count per generated query. Combined with `generation.min_tokens == generation.max_tokens` and `generation.ignore_eos: true`, this lets you pin exact input/output token lengths for like-for-like comparisons. |
| `synthetic.generation_concurrency` | Parallel LLM calls during generation (default `8`). Each completed query is streamed to disk under a write lock: a mid-generation failure preserves everything finished so far. Raise for fast endpoints, lower for rate-limited ones. |
| `synthetic.temperature` | Sampling temperature for the generator LLM (default `0.9`). |
| `synthetic.disable_thinking` | Default `true`. Injects `chat_template_kwargs: {enable_thinking: false}` into the request so reasoning models (Nemotron Omni, Qwen-Reasoning, …) skip chain-of-thought and return the question in `content`; otherwise CoT can exhaust the token budget and leave `content` empty. Set `false` only for non-reasoning endpoints. |
| `synthetic.extra_body` | Escape hatch: arbitrary keys merged into the LLM request body, for example `{top_p: 0.95, presence_penalty: 0.5, response_format: {type: json_object}}`. Merged after `disable_thinking`, so explicit keys win. |
| `synthetic.llm_url` | OpenAI-compatible endpoint used for generation. Typically the same NIM the RAG server proxies, but it can be any chat-completions endpoint. |
| `synthetic.llm_model` | Model name passed to the endpoint. Empty string → auto-discover from `/v1/models`. |
| `synthetic.prompts_file` | Custom YAML of prompt templates. `null` falls back to the bundled `prompts/default_prompts.yaml`. |
| `synthetic.jsonl_output_path` | Where the generated queries land. Re-running with the same path overwrites it. |

Because generated queries are persisted to disk, you can flip a synthetic-driven config to a file-driven one for subsequent runs by replacing the `synthetic` block with `file: <jsonl_output_path>`: useful for keeping the load identical while iterating on retrieval or reranker tuning.


## Configuration reference

Configuration is a YAML file validated by `rag_perf.config.RunConfig` (Pydantic v2). Top-level sections:

### Top-level

| Field | Type | Default | Purpose |
|---|---|---|---|
| `target` | object |: | Where the RAG server lives. |
| `aiperf` | object |: | Whether to run the aiperf load-test phase. |
| `load` | object |: | Load-generation parameters. |
| `rag` | object |: | RAG-specific request body fields forwarded to `/v1/generate`. |
| `generation` | object |: | LLM generation parameters (max_tokens, temperature, …). |
| `input` | object |: | Where queries come from. |
| `output` | object |: | Output directory and formats. |
| `model_name` | string | `nvidia/nemotron-3-super-120b-a12b` | Model name passed to aiperf using `-m`. Should match `APP_LLM_MODELNAME`. |
| `tokenizer` | string | `""` | Optional HuggingFace tokenizer ID for token counting; empty = use server-reported counts. |

### `target`

| Field | Type | Default | Purpose |
|---|---|---|---|
| `url` | string | `http://localhost:8081` | Base URL of the RAG server. |
| `timeout_s` | int | `300` | Per-request wall-clock timeout in seconds. |

### `aiperf`

| Field | Type | Default | Purpose |
|---|---|---|---|
| `enabled` | bool | `true` | When `false`, skip the aiperf load test (profile-only mode). |

### `load`

| Field | Type | Default | Purpose |
|---|---|---|---|
| `mode` | `concurrency` \| `request_rate` | `concurrency` | Load-generation strategy. |
| `concurrency` | int \| list[int] | `8` | Scalar = single point. List = sweep across that axis. |
| `request_rate` | float \| null | `null` | Target req/s when `mode=request_rate`. |
| `warmup_requests` | int | `10` | Requests sent (and discarded) before measurement. |
| `total_requests` | int | `200` | Total measured requests per point. |
| `duration_s` | float \| null | `null` | If set, run by wall-clock duration instead of request count. |
| `profile_requests` | int | `20` | Number of requests in the server-side profiling pass that runs before aiperf. |
| `iterations` | int | `1` | Repeat the full grid this many times. Each repetition writes to its own `iter_<i>/` subdirectory. |
| `sleep_between_points_s` | int | `0` | Seconds to sleep between grid points so the server can drain. `60` matches the blueprint pipeline's default. |

### `rag`

Forwarded verbatim to `POST /v1/generate`. Any field can be overridden per-query (see [Query inputs](#query-inputs)).

| Field | Type | Default | Purpose |
|---|---|---|---|
| `collection_names` | list[string] | `["default"]` | Vector-DB collection(s) to search. |
| `vdb_top_k` | int \| list[int] (each 1–400) | `100` | Chunks retrieved from the vector DB before reranking. Scalar = single value, list = sweep axis. |
| `reranker_top_k` | int \| list[int] (each 1–25) | `10` | Chunks passed to the LLM after reranking. Scalar = single value, list = sweep axis. |
| `enable_reranker` | bool | `true` | Whether to run the reranker stage. |
| `enable_citations` | bool | `true` | Whether the server returns citation chunks. |
| `use_knowledge_base` | bool | `true` | False = skip retrieval, send query bare to the LLM. |
| `confidence_threshold` | float (0–1) | `0.0` | Minimum relevance score for retrieved chunks. |

### `generation`

| Field | Type | Default | Purpose |
|---|---|---|---|
| `max_tokens` | int | `512` | Maximum output tokens. |
| `min_tokens` | int \| null | `null` | Minimum output tokens. Set equal to `max_tokens` to pin output length. |
| `ignore_eos` | bool | `false` | Pass `ignore_eos:true` to the inference backend; combine with `min_tokens` for fixed output length. |
| `temperature` | float (0–2) | `0.0` | Sampling temperature. |

### `input`

Set **exactly one** of `file` or `synthetic` (mutually exclusive). When both are unset, `synthetic` is auto-filled with defaults so a bare config still validates.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `file` | string \| null | `null` | Path to a query file. Format auto-detected from extension: `.jsonl` (one JSON object per line with a `query` key) or `.csv` (must have a `query` column). Mutually exclusive with `synthetic`. |
| `synthetic` | object \| null | `null` (auto-filled when `file` is also null) | LLM-generated synthetic queries. Fields below. Mutually exclusive with `file`. |
| `sampling` | string | `random` | Sampling strategy for the query list (`random`, `sequential`, `shuffle-once`). |
| `seed` | int | `42` | RNG seed for reproducible sampling. |
| `synthetic.mode` | `random` \| `dataset_based` | `random` | LLM-based generation strategy. |
| `synthetic.num_queries` | int | `50` | Distinct synthetic queries to generate. |
| `synthetic.min_query_tokens` | int | `50` | Approximate min token count per generated query. |
| `synthetic.llm_url` | string | `http://localhost:8999/v1/chat/completions` | OpenAI-compatible endpoint for generation. |
| `synthetic.llm_model` | string | `""` | Model name; empty = auto-discover from `/v1/models`. |
| `synthetic.prompts_file` | string \| null | `null` | Custom prompt templates YAML; null = use bundled defaults. |
| `synthetic.jsonl_output_path` | string | `./rag-perf-synthetic-queries.jsonl` | Where generated queries are written. |
| `synthetic.dataset_file` | string \| null | `null` | Explicit dataset file (required for `dataset_based`). |
| `synthetic.dataset_name` | string \| null | `null` | Dataset name for auto-lookup under `./datasets/`. |

### `output`

| Field | Type | Default | Purpose |
|---|---|---|---|
| `dir` | string | `./rag-perf-results` | Root output directory; a timestamped subdir is created per run. |
| `formats` | list[string] | `[json, csv]` | Export formats: `json`, `csv`, `jsonl_raw`. |
| `markdown_report` | bool | `true` | Write a Markdown summary (`report.md`). |
| `save_responses` | bool | `false` | Persist full generated text per request (large). |
| `cluster` | string | `""` | Cluster identifier stamped into artifact dir names. |
| `gpu` | string | `""` | GPU type stamped into artifact dir names. |
| `experiment_name` | string | `""` | Experiment label stamped into artifact dir names. |

## CLI surface

The CLI is intentionally minimal. The YAML is the single source of truth for behaviour; to vary a parameter, edit the file or copy it to a new one. This keeps every run reproducible from a single artefact you can commit to version control.

| Flag | Purpose |
|---|---|
| `-c / --config FILE` | Required. Path to the YAML config. |
| `--help` | Show usage and exit. |
| `--version` | Print the rag-perf version and exit. |


## Output layout

Every invocation creates a fresh timestamped directory `output.dir/run_<ts>/`. The contents depend on the run shape:

- **Single point + `aiperf.enabled=true`**: flat layout:
  ```
  run_<ts>/{report.md, results.csv, results.json, profiling/, aiperf_rag_on/}
  ```
- **Single point + `aiperf.enabled=false`**: flat, profile-only layout:
  ```
  run_<ts>/{profile_report.md, profile_results.json, profiling/}
  ```
- **Multiple points or `load.iterations > 1`**: nested layout:
  ```
  run_<ts>/
  ├── report.md, results.csv, results.json     # aggregate, one row per point
  └── iter_<i>/
      └── CR:<c>_ISL:<i>_OSL:<o>_VDB-K:<v>_RERANKER-K:<r>_Model:<m>[_Cluster:<c>][_GPU:<g>][_Experiment:<e>]/
          ├── profiling/
          └── aiperf_rag_on/
  ```


## Source layout

```
scripts/rag-perf/
├── pyproject.toml
├── configs/
│   ├── quick_profile.yaml      # profile-only preset
│   ├── single_run.yaml         # one concurrency, full report
│   └── sweep.yaml              # multi-axis sweep (concurrency / vdb_top_k / reranker_top_k as scalar or list)
├── examples/queries.jsonl       # sample query JSONL
├── prompts/default_prompts.yaml # synthetic-query prompt templates
└── rag_perf/
    ├── __init__.py             # public API re-exports
    ├── __main__.py             # python -m rag_perf entry point
    ├── config.py               # RunConfig + sub-models + the three enums
    ├── query.py                # QueryLoader + SyntheticQueryGenerator
    ├── runner.py               # RagProfiler + AiperfRunner + BenchmarkRunner.run
    ├── reporting.py            # MetricsAggregator + Reporter + result dataclasses
    ├── cli.py                  # Single Click command + startup banner
    └── plugin/                 # aiperf endpoint plugin (nvidia_rag)
```

Unit tests live under `tests/unit/test_rag_perf/` (run with `uv run --project scripts/rag-perf python -m pytest tests/unit/test_rag_perf/`).


## Related Topics

- [`scripts/eval/README.md`](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/scripts/eval/README.md): `evaluate_rag.py`, the runnable RAGAS-based accuracy benchmark CLI.
- [Evaluate Your NVIDIA RAG Blueprint System](evaluate.md): conceptual / notebook overview of the RAGAS metrics.
- [RAG Accuracy Benchmarks](accuracy-benchmarks.md): published accuracy results across datasets and configurations.
- [Best Practices for Common Settings](accuracy_perf.md): accuracy / performance tradeoff guidance.
- [NVIDIA RAG Blueprint Documentation](readme.md)
- Underlying load engine: [aiperf](https://github.com/NVIDIA/aiperf).

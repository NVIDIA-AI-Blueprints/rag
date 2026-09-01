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
Click-based CLI for ``rag-perf`` — also home to the startup banner.

Single entrypoint::

    rag-perf -c configs/<preset>.yaml

The YAML is the single source of truth for behaviour. The CLI accepts only
``--config`` (and ``--help`` / ``--version``); to vary a parameter, edit
the YAML or copy it to a new file. This keeps every run reproducible from
a single artefact.

YAML knobs that decide run shape:

- ``load.concurrency`` — scalar or list controls single-point vs sweep.
- ``aiperf.enabled``    — set False for profiling-only (no load test).
- ``sweep.{vdb_top_k, reranker_top_k}`` — optional cross-axes.
"""

from __future__ import annotations

import importlib.metadata

import click
from pydantic import ValidationError
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from rag_perf.config import RunConfig
from rag_perf.reporting import Reporter
from rag_perf.runner import BenchmarkRunner

_console = Console()


# ────────────────────────────────────────────────────────────────────────────
# Banner
# ────────────────────────────────────────────────────────────────────────────

# ASCII block-letter logo — spells "RAG PERF", 65 chars wide.
_LOGO_LINES = [
    " ██████╗  █████╗  ██████╗    ██████╗ ███████╗██████╗ ███████╗",
    " ██╔══██╗██╔══██╗██╔════╝    ██╔══██╗██╔════╝██╔══██╗██╔════╝",
    " ██████╔╝███████║██║  ███╗   ██████╔╝█████╗  ██████╔╝█████╗  ",
    " ██╔══██╗██╔══██║██║   ██║   ██╔═══╝ ██╔══╝  ██╔══██╗██╔══╝  ",
    " ██║  ██║██║  ██║╚██████╔╝   ██║     ███████╗██║  ██║██║     ",
    " ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝    ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝     ",
]
_LOGO_COLORS = ["#76B900", "#55CC22", "#00CC88", "#00BEB4", "#00AADD", "#0099FF"]


def print_banner(console: Console | None = None) -> None:
    """Print the green-gradient RAG PERF ASCII logo followed by a subtitle line."""
    con = console or Console()

    try:
        version = importlib.metadata.version("nvidia-rag-perf")
    except importlib.metadata.PackageNotFoundError:
        version = "dev"

    logo = Text()
    for line, color in zip(_LOGO_LINES, _LOGO_COLORS, strict=False):
        logo.append(line + "\n", style=f"bold {color}")

    con.print()
    con.print(logo)
    con.rule(style="bold cyan")
    con.print(
        f"  [bold white]NVIDIA RAG Blueprint[/bold white]  [dim]·[/dim]  "
        f"[cyan]Performance Benchmarking[/cyan]  [dim]·[/dim]  "
        f"[dim]v{version}[/dim]"
    )
    con.rule(style="bold cyan")
    con.print()


# ────────────────────────────────────────────────────────────────────────────
# CLI logic
# ────────────────────────────────────────────────────────────────────────────


def _load_config(config_file: str) -> RunConfig:
    """Load and validate a ``RunConfig`` from *config_file*.

    On YAML / validation errors, prints a human-readable list of the failing
    fields and aborts. The CLI is config-only, so there are no programmatic
    overrides at this layer.
    """
    try:
        return RunConfig.from_yaml(config_file)
    except FileNotFoundError as exc:
        _console.print(
            f"\n  [bold red]✘  Config file not found:[/bold red]  [cyan]{config_file}[/cyan]\n"
            f"  Check the path and try again.\n"
        )
        raise click.Abort() from exc
    except ValidationError as exc:
        _console.print(
            f"\n  [bold red]✘  Configuration errors in[/bold red] [cyan]{config_file}[/cyan]:\n"
        )
        for err in exc.errors():
            loc = " → ".join(str(p) for p in err["loc"])
            _console.print(f"    [red]•[/red]  [bold]{loc}[/bold]  —  {err['msg']}")
        _console.print(
            "\n  Fix the above fields in your config YAML and try again.\n"
        )
        raise click.Abort() from exc


def _print_run_info(cfg: RunConfig) -> None:
    """Print a concise summary of the run configuration before starting."""
    _console.rule("[bold cyan]RAG-Perf[/bold cyan]")
    source_info = (
        f"[cyan]{cfg.input.file}[/cyan]"
        if cfg.input.source.value in ("jsonl", "csv")
        else f"[cyan]synthetic ({cfg.input.synthetic.mode.value})[/cyan] → "
        f"[dim]{cfg.input.synthetic.jsonl_output_path}[/dim]"
    )
    cs = cfg.load.concurrency_list
    conc_label = str(cs[0]) if len(cs) == 1 else ", ".join(str(c) for c in cs)
    grid_axes = []
    if isinstance(cfg.rag.vdb_top_k, list):
        grid_axes.append(f"vdb_top_k={cfg.rag.vdb_top_k}")
    if isinstance(cfg.rag.reranker_top_k, list):
        grid_axes.append(f"reranker_top_k={cfg.rag.reranker_top_k}")
    grid_extra = "  " + "  ".join(grid_axes) if grid_axes else ""
    iter_extra = (
        f"  Iterations: [cyan]{cfg.load.iterations}[/cyan]"
        if cfg.load.iterations > 1
        else ""
    )
    aiperf_label = "on" if cfg.aiperf.enabled else "off (profile-only)"

    _console.print(
        f"  Target:       [cyan]{cfg.target.url}[/cyan]\n"
        f"  Collection:   [cyan]{', '.join(cfg.rag.collection_names)}[/cyan]\n"
        f"  vdb_top_k:    [cyan]{cfg.rag.vdb_top_k}[/cyan]  "
        f"reranker_top_k: [cyan]{cfg.rag.reranker_top_k}[/cyan]\n"
        f"  Input:        {source_info}\n"
        f"  Concurrency:  [cyan]{conc_label}[/cyan]  "
        f"Requests/point: [cyan]{cfg.load.total_requests}[/cyan]"
        f"{iter_extra}"
        f"{grid_extra}\n"
        f"  aiperf:       [cyan]{aiperf_label}[/cyan]"
    )
    _console.rule()

    # Full resolved config dump — every field that drives the run, including
    # the synthetic-generation block. Printed verbatim as YAML so users can
    # diff runs, paste it into bug reports, or copy any section into a new
    # config file. Comes from the same to_yaml_str() the markdown report uses.
    _console.print("[bold]Resolved configuration:[/bold]")
    _console.print(
        Syntax(
            cfg.to_yaml_str(),
            "yaml",
            theme="ansi_dark",
            background_color="default",
            word_wrap=True,
        )
    )
    _console.rule()


# ────────────────────────────────────────────────────────────────────────────
# Click command
# ────────────────────────────────────────────────────────────────────────────


@click.command()
@click.version_option(package_name="nvidia-rag-perf", prog_name="rag-perf")
@click.option(
    "--config",
    "-c",
    "config_file",
    required=True,
    metavar="FILE",
    help="Path to the YAML config file (required).",
)
def main(config_file: str) -> None:
    """rag-perf — RAG Server Performance Benchmarking Tool.

    Drives the deployed RAG server with traffic described entirely by the
    --config YAML, captures server-side stage timing plus (optionally)
    aiperf load-test metrics, and writes a unified report.

    \b
    Examples:
        rag-perf -c configs/quick_profile.yaml
        rag-perf -c configs/single_run.yaml
        rag-perf -c configs/sweep.yaml
    """
    cfg = _load_config(config_file)

    print_banner(_console)
    _print_run_info(cfg)

    summaries = BenchmarkRunner.run(cfg)
    if not summaries:
        _console.print("[red]✗ Benchmark produced no summaries.[/red]")
        raise click.exceptions.Exit(1)

    if len(summaries) == 1:
        title = (
            "RAG-Perf Profile" if not cfg.aiperf.enabled else "RAG-Perf Results"
        )
        Reporter.print_summary(summaries[0], title=title)
    else:
        # Multi-summary runs (sweep across any axis, n_times>1, or both):
        # print the rich per-point summary table for every point so users see
        # the full stage breakdown / citation quality / bottleneck for each,
        # then the aggregate comparison table at the end.
        for s in summaries:
            label_parts: list[str] = []
            if s.concurrency is not None:
                label_parts.append(f"conc={s.concurrency}")
            if s.vdb_top_k is not None:
                label_parts.append(f"vdb_top_k={s.vdb_top_k}")
            if s.reranker_top_k is not None:
                label_parts.append(f"rr_top_k={s.reranker_top_k}")
            label = "  ".join(label_parts) or "point"
            point_title = (
                f"RAG-Perf Profile — {label}"
                if not cfg.aiperf.enabled
                else f"RAG-Perf Results — {label}"
            )
            Reporter.print_summary(s, title=point_title)
        Reporter.print_sweep_table(summaries)

    # Bug E: if every profiling request failed across all points (e.g. the
    # server URL was wrong or the service was down), the run is meaningless.
    # Surface that as a non-zero exit so CI doesn't silently pass.
    failed = sum(s.profile_requests_failed for s in summaries)
    total = sum(s.profile_requests_total for s in summaries)
    if total > 0 and failed == total:
        _console.print(
            f"[bold red]✗ All {total} profiling requests failed across "
            f"{len(summaries)} point(s).[/bold red]"
        )
        raise click.exceptions.Exit(1)

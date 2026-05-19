#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor environment provider for Brev VMs (GPU-only, ephemeral per CI run).

CPU evals use envs.local_env:LocalEnvironment (runner == target). This
module is engaged when the workflow sets $BREV_INSTANCE=rag-eval-gpu-<uuid>
for a GPU eval; the named-pool `rag-eval-target` model is gone.

Lifecycle:
    start()    → provision fresh VM, clone repo, smoke-test exec
    exec()     → brev exec <instance> -- <command>
    upload()   → tar | base64 | brev exec (brev copy has nesting bugs)
    download() → brev exec | base64 -d | tar
    stop()     → no-op (ci/run_skill_eval.sh trap owns teardown)

Provision picks the cheapest cloud type matching task.toml [metadata]
GPU requirements via `brev search --json`, then `brev create` with the
type fed via stdin. Trials within one eval share a deploy (step-1
deploys, step-2 probes), so stop() never tears down — the script's
EXIT trap handles brev delete/stop after both trials finish.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shlex
import subprocess
import uuid
from enum import Enum
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult

logger = logging.getLogger(__name__)

DEFAULT_INSTANCE = os.environ.get("BREV_INSTANCE")
BREV_EXEC_TIMEOUT = int(os.environ.get("BREV_EXEC_TIMEOUT", "1800"))
BREV_COPY_TIMEOUT = int(os.environ.get("BREV_COPY_TIMEOUT", "300"))
BREV_CREATE_TIMEOUT = int(os.environ.get("BREV_CREATE_TIMEOUT", "600"))


class BrevEnvironmentType(str, Enum):
    BREV = "brev"


class BrevEnvironment(BaseEnvironment):
    """Harbor environment that drives a remote Brev VM via the brev CLI."""

    def __init__(self, **kwargs):  # noqa: ANN003
        super().__init__(**kwargs)
        self._instance_name: str | None = None
        self._created_by_us = False  # for stop() — only delete what we created
        self._started = False

    @staticmethod
    def type() -> BrevEnvironmentType:
        return BrevEnvironmentType.BREV

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def supports_gpus(self) -> bool:
        return True

    @property
    def can_disable_internet(self) -> bool:
        return False

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def _validate_definition(self) -> None:
        if not _which("brev"):
            raise RuntimeError(
                "brev CLI not found on PATH. Install per https://docs.brev.dev/"
            )

    def _read_task_metadata(self) -> dict:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]
        task_toml = self.environment_dir.parent / "task.toml"
        if not task_toml.exists():
            return {}
        return tomllib.loads(task_toml.read_text()).get("metadata", {}) or {}

    async def start(self, force_build: bool) -> None:
        if self._started:
            return

        meta = self._read_task_metadata()
        # Always ephemeral: the workflow passes $BREV_INSTANCE=rag-eval-gpu-<uuid>
        # per CI run (fresh each invocation), or we auto-generate. The
        # script (ci/run_skill_eval.sh) pre-provisions the VM out-of-band
        # before invoking harbor — necessary because harbor's environment-
        # start timeout is hardcoded to 600s in trial.py, but cold H100×2
        # provisioning routinely takes 10-15 min on Brev. We mirror VSS's
        # pattern: script provisions, harbor inherits a ready VM.
        self._instance_name = (
            DEFAULT_INSTANCE
            or meta.get("brev_instance")
            or f"rag-harbor-{uuid.uuid4().hex[:8]}"
        )
        existing = await _find_brev_instance(self._instance_name)
        if existing and existing.get("status") == "RUNNING":
            logger.info(
                "Brev target: %s (pre-provisioned by script)",
                self._instance_name,
            )
            self._created_by_us = False  # script owns lifecycle
        else:
            logger.info(
                "Brev target: %s (no pre-provision — provisioning now; "
                "expect timeout if harbor's 600s start budget is short for "
                "the platform)",
                self._instance_name,
            )
            await self._provision(meta)
            self._created_by_us = True

        # Smoke test: confirm we can exec on the instance.
        result = await _run_brev_exec(
            self._instance_name, "echo harbor-ready", timeout=60,
        )
        if result.return_code != 0 or "harbor-ready" not in (result.stdout or ""):
            raise RuntimeError(
                f"Cannot reach Brev instance '{self._instance_name}': "
                f"rc={result.return_code} stderr={result.stderr!r}"
            )

        # Hardware validation against task.toml [metadata] floors. Each
        # check is a no-op if its key is absent — current CPU evals
        # (nvidia-hosted) declare none, so this is fully passthrough until
        # we add a GPU eval (vlm.json, helm_h100.json, etc.). When GPU
        # evals land, this catches "wrong instance type" / "insufficient
        # VRAM" / "old driver" in ~2s instead of in a confused docker
        # compose failure ~5 min into the deploy.
        req = {
            "gpu_type": meta.get("gpu_type"),
            "gpu_count": meta.get("gpu_count"),
            "min_vram_gb_per_gpu": meta.get("min_vram_gb_per_gpu"),
            "min_root_disk_gb": meta.get("min_root_disk_gb"),
            "min_gpu_driver_version": meta.get("min_gpu_driver_version"),
        }
        if any(req.values()):
            inst_json = await _find_brev_instance_json(self._instance_name)
            if inst_json:
                _check_instance_matches(inst_json, req)
            else:
                logger.warning(
                    "Couldn't fetch JSON instance metadata for %s — skipping "
                    "static GPU validation; live checks still run.",
                    self._instance_name,
                )
            await _check_live_resources(self._instance_name, req)

        # Pre-create the harbor-expected directories with the user's ownership
        # so the trial agent and verifier can write to them without sudo.
        # LocalEnvironment pre-creates logs/agent/sessions, logs/verifier,
        # logs/artifacts under its workdir. We mirror that on the target so
        # tests/test.sh (which writes $VERIFIER_DIR/reward.txt with default
        # $VERIFIER_DIR=/logs/verifier) finds the dir already present —
        # otherwise uvx errors writing to a non-existent path and Harbor
        # downloads an empty verifier dir → RewardFileNotFoundError.
        await _run_brev_exec(
            self._instance_name,
            "sudo mkdir -p /logs/agent/sessions /logs/verifier "
            "/logs/artifacts /tests /solution /skills /installed-agent && "
            'sudo chown -R "$(whoami):$(id -gn)" '
            "/logs /tests /solution /skills /installed-agent",
            timeout=60,
        )

        # Get the repo onto the target at $HOME/rag via git clone (VSS
        # pattern). Harbor only uploads task-local files (skills/, tests/),
        # not the surrounding repo. A fresh clone per run wipes any stale
        # files from prior trials on warm-pool VMs — no orphan-file class
        # of bugs to worry about. Trade-off: we test what's been pushed to
        # the branch, not uncommitted runner state. For PR-driven CI that's
        # always already pushed, this is fine.
        await self._clone_repo()

        self._started = True

    async def _clone_repo(self) -> None:
        """Fresh `git clone` of the eval branch into $HOME/rag on target.

        Mirrors VSS's pattern. `rm -rf` first so warm-pool runs don't
        accumulate orphan files. Shallow clone (--depth 1) for speed.

        Branch comes from $EVAL_TARGET_BRANCH (set by run_skill_eval.sh
        from `git rev-parse --abbrev-ref HEAD` on the runner — i.e. the
        ref that GitHub's `actions/checkout` materialised). Repo URL
        defaults to the upstream blueprint; override with $RAG_REPO_URL.
        For private repos, pass $GITHUB_TOKEN — embedded into the URL.
        """
        assert self._instance_name
        branch = os.environ.get("EVAL_TARGET_BRANCH", "main")
        repo_url = os.environ.get(
            "RAG_REPO_URL",
            "https://github.com/NVIDIA-AI-Blueprints/rag.git",
        )
        gh_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

        # Token-embedded URL for private repos. Public repo works without.
        clone_url = repo_url
        if gh_token and repo_url.startswith("https://github.com/"):
            clone_url = repo_url.replace(
                "https://", f"https://x-access-token:{gh_token}@",
            )

        logger.info("Cloning %s @ %s → %s:$HOME/rag",
                    repo_url, branch, self._instance_name)

        # `sudo` because docker-compose volumes (e.g. milvus/rdb_data_meta_kv/*)
        # are owned by root — Milvus's container writes them as root via the
        # bind mount, and the ubuntu user can't rm them. Without sudo the
        # rm errors out before git clone, the agent never sees a repo, and
        # both trials raise RuntimeError in start().
        clone_cmd = (
            f'sudo rm -rf "$HOME/rag" && '
            f"git clone --depth 1 --branch {shlex.quote(branch)} "
            f"{shlex.quote(clone_url)} \"$HOME/rag\""
        )
        result = await _run_brev_exec(
            self._instance_name, clone_cmd, timeout=BREV_COPY_TIMEOUT,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"git clone on {self._instance_name} failed: {result.stderr}"
            )

        # Forward env vars that the deploy needs into ~/.eval_env on the
        # target. The runner's shell has these (CI Pipeline sets them);
        # without forwarding, the agent's fresh shell on the target
        # doesn't see them and Milvus tries to use the (non-existent) GPU.
        env_lines = [
            'export RAG_REPO_ROOT="$HOME/rag"',
            # NVIDIA-hosted CPU mode — these force Milvus off GPU code paths.
            'export APP_VECTORSTORE_ENABLEGPUSEARCH=False',
            'export APP_VECTORSTORE_ENABLEGPUINDEX=False',
            # NGC token forwarded so `docker login nvcr.io` works on the target
            # when the skill needs to pull NIM containers.
        ]
        # NGC for docker login; NVIDIA_* for cloud NIM auth; the rest are
        # for the verifier (tests/test.sh → generic_judge.py), which uses
        # `set -uo pipefail` and dies on the first unset judge var. On
        # LocalEnvironment these come from the runner's shell; on Brev
        # the target only sees what we explicitly forward here.
        for var in (
            "NGC_API_KEY",
            "NVIDIA_API_KEY",
            "NVIDIA_INFERENCE_KEY",
            "JUDGE_ANTHROPIC_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "ANTHROPIC_MODEL",
            "JUDGE_FULL_MODEL",
            "JUDGE_MODEL",
            "CLAUDE_CODE_DISABLE_THINKING",
        ):
            val = os.environ.get(var)
            if val:
                env_lines.append(f'export {var}={shlex.quote(val)}')

        env_block = "\n".join(env_lines)
        bootstrap = (
            f'cat > "$HOME/.eval_env" <<\'__HARBOR_EOF__\'\n'
            f"{env_block}\n"
            f"__HARBOR_EOF__\n"
            f'if ! grep -q "source.*\\.eval_env" "$HOME/.profile" 2>/dev/null; then '
            f'  echo "source ~/.eval_env 2>/dev/null" >> "$HOME/.profile"; '
            f"fi\n"
            f"ls \"$HOME/rag/deploy/compose/\" | head -5"
        )
        result = await _run_brev_exec(self._instance_name, bootstrap, timeout=30)
        if result.return_code != 0:
            raise RuntimeError(
                f"Repo stage verification failed: {result.stderr}"
            )
        logger.info("Repo staged + env forwarded. Sample:\n%s", result.stdout)

        # Install uv on the target if missing. The verifier (tests/test.sh)
        # invokes `uvx --with anthropic,claude-agent-sdk python ...` to run
        # the LLM-as-judge. Without uv, test.sh errors with
        # `uvx: command not found` and writes no /logs/verifier/reward.txt,
        # so Harbor's reward-read step fails with RewardFileNotFoundError.
        # Idempotent: only installs if uvx isn't already on PATH (warm pool).
        uv_install = await _run_brev_exec(
            self._instance_name,
            'if ! command -v uvx >/dev/null 2>&1 && ! [ -x "$HOME/.local/bin/uvx" ]; then '
            '  curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -2; '
            'fi; '
            '"$HOME/.local/bin/uvx" --version 2>&1 | head -1',
            timeout=180,
        )
        if uv_install.return_code != 0:
            logger.warning("uv install/check failed on target: %s",
                           uv_install.stderr)
        else:
            logger.info("uvx on target: %s", uv_install.stdout.strip())

        # Pre-authenticate docker on the target so the agent's
        # `docker compose up` can pull from nvcr.io.
        ngc = os.environ.get("NGC_API_KEY")
        if ngc:
            login_result = await _run_brev_exec(
                self._instance_name,
                f"echo {shlex.quote(ngc)} | "
                "docker login nvcr.io -u '$oauthtoken' --password-stdin",
                timeout=60,
            )
            if login_result.return_code != 0:
                logger.warning("docker login nvcr.io failed on target: %s",
                               login_result.stderr)

    async def _provision(self, meta: dict) -> None:
        """brev create — type from task.toml [metadata] or catalog search.

        Type resolution:
          - `brev_type` in metadata → use directly (CPU + explicit GPU specs).
          - else → `brev search --json` cheapest match on gpu_type /
            gpu_count / min_vram_gb_per_gpu / min_root_disk_gb.

        Type is fed via stdin (not `--type`). The `--type` path still
        reads stdin as a fallback type list — that caused the stdin-leak
        bug in run 25925485685 where harbor's `find` loop left extra
        instance names on stdin. VSS uses stdin exclusively; we mirror.
        """
        brev_type = meta.get("brev_type")
        if not brev_type:
            requirements = {
                "brev_search": meta.get("brev_search") or meta.get("gpu_type"),
                "gpu_type": meta.get("gpu_type"),
                "gpu_count": int(meta.get("gpu_count") or 1),
                "min_vram_gb_per_gpu": int(meta.get("min_vram_gb_per_gpu") or 0),
                "min_root_disk_gb": int(meta.get("min_root_disk_gb") or 0),
            }
            brev_type = await _find_cheapest_matching_type(requirements)
            if not brev_type:
                raise RuntimeError(
                    "Cannot provision: no Brev cloud type matches "
                    f"requirements {requirements}. Set `brev_type` in "
                    "task.toml [metadata] explicitly or adjust gpu_type / "
                    "min_vram_gb_per_gpu."
                )
            logger.info("Resolved cheapest matching type: %s", brev_type)

        logger.info("Creating Brev instance %s type=%s",
                    self._instance_name, brev_type)
        create = await _run_brev(
            "create", self._instance_name, "--detached",
            stdin_data=brev_type,
            timeout=BREV_CREATE_TIMEOUT,
        )
        if create.return_code != 0:
            raise RuntimeError(
                f"brev create {self._instance_name} (type={brev_type}) "
                f"failed: {create.stderr}"
            )
        await _wait_for_running(self._instance_name)

    # -----------------------------------------------------------------------
    # Teardown
    # -----------------------------------------------------------------------

    async def stop(self, delete: bool) -> None:
        """No-op — ci/run_skill_eval.sh's EXIT trap owns teardown.

        Trials within one eval share a deploy (step-1 deploys, step-2
        probes), so the VM must survive across step boundaries even
        though Harbor passes delete=True between them. The script trap
        routes to `brev delete` or `brev stop` after both trials finish,
        based on the provider's lifecycle. Matches VSS pattern exactly.
        """
        if not self._instance_name:
            return
        logger.info(
            "Leaving Brev instance %s running — script trap owns teardown "
            "(Harbor delete=%s ignored)",
            self._instance_name, delete,
        )
        self._started = False

    # -----------------------------------------------------------------------
    # File transfer
    # -----------------------------------------------------------------------

    async def upload_file(
        self, source_path: Path | str, target_path: str,
    ) -> None:
        """Copy a single file to the target via `brev copy`.

        Earlier impl base64-encoded the file and passed it as a CLI arg —
        any file over ~64 KB blew past Linux ARG_MAX. `brev copy` handles
        arbitrary sizes and is reliable for single files.
        """
        assert self._instance_name
        src = Path(source_path)
        target_dir = str(Path(target_path).parent) or "/"
        # Ensure the target dir exists with the right ownership BEFORE copy.
        mkdir = await _run_brev_exec(
            self._instance_name,
            f"sudo mkdir -p {shlex.quote(target_dir)} && "
            f'sudo chown "$(whoami):$(id -gn)" {shlex.quote(target_dir)}',
            timeout=60,
        )
        if mkdir.return_code != 0:
            raise RuntimeError(f"upload_file mkdir failed: {mkdir.stderr}")
        copy = await _run_brev(
            "copy", str(src), f"{self._instance_name}:{target_path}",
            timeout=BREV_COPY_TIMEOUT,
        )
        if copy.return_code != 0:
            raise RuntimeError(f"upload_file copy failed: {copy.stderr}")

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str,
    ) -> None:
        """Copy a directory tree to the target.

        Tarball goes to a local /tmp file → `brev copy` to target → untar.
        Earlier impl base64-encoded the tarball into a `brev exec` argv,
        which capped uploads at the OS ARG_MAX (~128 KB) — Harbor's
        post-trial agent trajectory upload routinely exceeds that, so the
        trial failed before the verifier ever ran.
        """
        assert self._instance_name
        src = str(source_dir).rstrip("/")
        local_tar = Path(f"/tmp/harbor-up-{uuid.uuid4().hex[:8]}.tar.gz")
        try:
            subprocess.check_call(
                ["tar", "-czf", str(local_tar), "-C", src, "."], timeout=180,
            )
            remote_tar = f"/tmp/harbor-up-{uuid.uuid4().hex[:8]}.tar.gz"
            # Ensure target_dir exists with proper ownership.
            mkdir = await _run_brev_exec(
                self._instance_name,
                f"sudo mkdir -p {shlex.quote(target_dir)} && "
                f'sudo chown "$(whoami):$(id -gn)" {shlex.quote(target_dir)}',
                timeout=60,
            )
            if mkdir.return_code != 0:
                raise RuntimeError(f"upload_dir mkdir failed: {mkdir.stderr}")
            copy = await _run_brev(
                "copy", str(local_tar), f"{self._instance_name}:{remote_tar}",
                timeout=BREV_COPY_TIMEOUT,
            )
            if copy.return_code != 0:
                raise RuntimeError(f"upload_dir copy failed: {copy.stderr}")
            extract = await _run_brev_exec(
                self._instance_name,
                f"tar -xzf {shlex.quote(remote_tar)} -C {shlex.quote(target_dir)} && "
                f"rm -f {shlex.quote(remote_tar)}",
                timeout=BREV_COPY_TIMEOUT,
            )
            if extract.return_code != 0:
                raise RuntimeError(f"upload_dir untar failed: {extract.stderr}")
        finally:
            local_tar.unlink(missing_ok=True)

    async def download_file(
        self, source_path: str, target_path: Path | str,
    ) -> None:
        assert self._instance_name
        marker = f"__HARBOR_B64_{uuid.uuid4().hex[:8]}__"
        result = await _run_brev_exec(
            self._instance_name,
            f"echo '{marker}START'; "
            f"base64 -w 0 {shlex.quote(source_path)}; "
            f"echo; echo '{marker}END'",
            timeout=BREV_COPY_TIMEOUT,
        )
        if result.return_code != 0:
            raise RuntimeError(f"download_file failed: {result.stderr}")
        raw = _extract_between_markers(result.stdout or "", marker)
        Path(target_path).write_bytes(base64.b64decode(raw))

    async def download_dir(
        self, source_dir: str, target_dir: Path | str,
    ) -> None:
        assert self._instance_name
        marker = f"__HARBOR_B64_{uuid.uuid4().hex[:8]}__"
        result = await _run_brev_exec(
            self._instance_name,
            f"echo '{marker}START'; "
            f"tar -czf - -C {shlex.quote(source_dir)} . 2>/dev/null | base64 -w 0; "
            f"echo; echo '{marker}END'",
            timeout=BREV_COPY_TIMEOUT,
        )
        if result.return_code != 0:
            raise RuntimeError(f"download_dir failed: {result.stderr}")
        raw = _extract_between_markers(result.stdout or "", marker)
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "-xzf", "-", "-C", str(target)],
            input=base64.b64decode(raw), check=True, timeout=120,
        )

    # -----------------------------------------------------------------------
    # exec
    # -----------------------------------------------------------------------

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        assert self._instance_name

        parts = [
            'export PATH="$HOME/.local/bin:$HOME/.claude/bin:$PATH";',
            "source ~/.profile 2>/dev/null;",
        ]
        if env:
            for k, v in env.items():
                parts.append(f"export {shlex.quote(k)}={shlex.quote(v)};")
        if cwd:
            parts.append(f"cd {shlex.quote(cwd)};")
        parts.append(command)
        inner = " ".join(parts)

        # Package manager install actions need sudo (brev exec runs as
        # the ubuntu user by default).
        needs_root = (
            user in ("root", 0)
            or bool(re.search(
                r"\b(apt-get|apt|apk|yum|dnf)\s+(install|add|update|upgrade)\b",
                command,
            ))
        )
        full = f"sudo bash -c {shlex.quote(inner)}" if needs_root else inner
        return await _run_brev_exec(
            self._instance_name, full, timeout=timeout_sec or BREV_EXEC_TIMEOUT,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _which(cmd: str) -> bool:
    return subprocess.run(
        ["which", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _extract_between_markers(stdout: str, marker: str) -> str:
    """Pull base64 payload from stdout between START/END sentinels.

    `brev exec` interleaves spinner / connection noise into stdout. We
    wrap the actual payload between unique markers and strip everything
    that isn't a valid base64 character on the way out.
    """
    m = re.search(rf"{marker}START\s*\n(.*?)\n{marker}END", stdout, re.DOTALL)
    if not m:
        raise RuntimeError(
            f"markers not found in brev exec output (len={len(stdout)})"
        )
    return "".join(
        c for c in m.group(1)
        if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    )


async def _run_brev(
    *args: str, timeout: int = 30, stdin_data: str | None = None,
) -> ExecResult:
    """Invoke `brev <args>` as a subprocess and return its result.

    Stdin handling:
      - default (stdin_data=None) → write a single newline. The CLI's
        interactive walkthrough reads stdin; supplying empty input makes
        it bail cleanly instead of hanging on a TTY prompt. We don't use
        DEVNULL because `brev create` reads its instance type from stdin
        as the canonical input path (per VSS pattern).
      - stdin_data="<type>" → fed to `brev create`'s instance-type prompt.
        Avoids the `--type` flag stdin-inherit bug seen in run 25925485685
        where harbor's `find` loop left extra lines on stdin and brev
        parsed them as bogus fallback types.
    """
    proc = await asyncio.create_subprocess_exec(
        "brev", *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(input=(stdin_data or "").encode() + b"\n"),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return ExecResult(stdout="", stderr=f"brev {args!r} timed out", return_code=124)
    return ExecResult(
        stdout=stdout_b.decode(errors="replace"),
        stderr=stderr_b.decode(errors="replace"),
        return_code=proc.returncode or 0,
    )


async def _run_brev_exec(instance: str, command: str, timeout: int) -> ExecResult:
    """`brev exec <instance> <command>` — command must be a SINGLE arg.

    Brev's CLI parses every positional arg after the first as another
    instance name. Passing `--` or `bash -c <cmd>` as separate args makes
    it try to SSH into "--", "bash", "-c" as instance names. Internally
    brev wraps the single command string in `bash -c` already.
    """
    return await _run_brev("exec", instance, command, timeout=timeout)


async def _find_brev_instance(name: str) -> dict | None:
    """Return the instance row from `brev ls` or None if not found.

    `brev ls` is human-formatted (no --json flag in current CLI).
    Column layout: NAME STATUS BUILD SHELL ID MACHINE
    """
    result = await _run_brev("ls", timeout=30)
    if result.return_code != 0:
        return None
    for line in (result.stdout or "").splitlines():
        cols = line.split()
        if len(cols) >= 4 and cols[0] == name:
            return {
                "name": cols[0],
                "status": cols[1],   # RUNNING / STOPPED / DEPLOYING / ...
                "build": cols[2],    # COMPLETED / BUILDING / ...
                "shell": cols[3],    # READY / NOT_READY / ...
            }
    return None


async def _find_brev_instance_json(name: str) -> dict | None:
    """Return the full instance row from `brev ls --json` or None.

    Used only for hardware validation (gpu_name, gpu_count, gpu_memory_gb,
    machine_type). The text `brev ls` doesn't expose those columns.

    Tolerates missing/old CLI by returning None on any parse failure —
    caller skips static validation and falls back to live checks.
    """
    result = await _run_brev("ls", "--json", timeout=30)
    if result.return_code != 0:
        return None
    try:
        payload = json.loads(result.stdout or "null")
    except json.JSONDecodeError:
        return None
    # Older CLIs return a bare list; newer wrap in {"instances": [...]}.
    rows = payload if isinstance(payload, list) else payload.get("instances", [])
    for row in rows or []:
        if isinstance(row, dict) and row.get("name") == name:
            return row
    return None


def _version_lt(a: str, b: str) -> bool:
    """Loose version compare: True if a < b. Dotted ints; ignores suffixes."""
    def _parts(s: str) -> list[int]:
        out: list[int] = []
        for tok in re.split(r"[^\d]+", s or ""):
            if tok:
                out.append(int(tok))
        return out
    pa, pb = _parts(a), _parts(b)
    n = max(len(pa), len(pb))
    pa += [0] * (n - len(pa))
    pb += [0] * (n - len(pb))
    return pa < pb


def _check_instance_matches(instance: dict, req: dict) -> None:
    """Validate that a Brev instance row meets task requirements.

    Reads from instance row (from `brev ls --json`):
      - gpu_name (e.g. "H100-SXM-80GB" — substring-matched)
      - gpu_count (int)
      - gpu_memory_gb (per-GPU VRAM)

    Compares against req dict from task.toml [metadata]:
      - gpu_type — substring match, case-insensitive ("H100" matches "H100-SXM-80GB")
      - gpu_count — instance must have at least this many
      - min_vram_gb_per_gpu — instance must have at least this much per GPU

    Each check skipped if its req key is absent. If the instance JSON is
    missing the matching field (older brev CLI, registered node), log
    warning and skip — live checks will catch real mismatches.

    Raises RuntimeError with a clear message on the first concrete mismatch.
    """
    name = instance.get("name") or "<unnamed>"

    req_gpu_type = req.get("gpu_type")
    if req_gpu_type:
        inst_gpu = instance.get("gpu_name")
        if inst_gpu is None:
            logger.warning(
                "Instance %s has no 'gpu_name' in JSON — skipping static "
                "gpu_type check (live validation still runs).", name,
            )
        elif req_gpu_type.lower() not in (inst_gpu or "").lower():
            raise RuntimeError(
                f"Brev instance '{name}' has gpu_name={inst_gpu!r}, "
                f"task requires gpu_type containing {req_gpu_type!r}"
            )

    req_count = int(req.get("gpu_count") or 0)
    if req_count:
        inst_count = int(instance.get("gpu_count") or 0)
        if inst_count and inst_count < req_count:
            raise RuntimeError(
                f"Brev instance '{name}' has gpu_count={inst_count}, "
                f"task requires at least {req_count}"
            )

    req_vram = int(req.get("min_vram_gb_per_gpu") or 0)
    if req_vram:
        inst_vram = int(instance.get("gpu_memory_gb") or 0)
        if inst_vram and inst_vram < req_vram:
            raise RuntimeError(
                f"Brev instance '{name}' has gpu_memory_gb={inst_vram} per GPU, "
                f"task requires at least {req_vram}"
            )


async def _check_live_resources(instance_name: str, req: dict) -> None:
    """Probe the target VM for disk + GPU driver against task floors.

    Each check skipped if its req key is absent. Raises RuntimeError on
    mismatch with the actual probed value in the message.

    Catches provider quirks `brev ls --json` doesn't surface — e.g. some
    providers list disk_min_gb=1600 but mount the big volume on
    /ephemeral, leaving / at ~100 GB → docker pull OOMs on NIM images.
    """
    min_disk = int(req.get("min_root_disk_gb") or 0)
    if min_disk:
        r = await _run_brev_exec(
            instance_name,
            "df -BG / | tail -1 | awk '{print $2}' | tr -d 'G'",
            timeout=60,
        )
        if r.return_code == 0:
            try:
                disk_gb = int((r.stdout or "0").strip())
            except ValueError:
                disk_gb = 0
            if disk_gb and disk_gb < min_disk:
                raise RuntimeError(
                    f"Brev instance '{instance_name}' root disk is {disk_gb}G, "
                    f"task requires at least {min_disk}G (NIM images need this)"
                )

    min_driver = req.get("min_gpu_driver_version")
    if min_driver:
        r = await _run_brev_exec(
            instance_name,
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 "
            "| head -1",
            timeout=60,
        )
        if r.return_code == 0:
            actual_lines = (r.stdout or "").strip().splitlines()
            actual = actual_lines[0].strip() if actual_lines else ""
            # nvidia-smi missing → message contains 'command not found' or similar
            looks_like_version = bool(re.match(r"^\d+\.\d+", actual))
            if looks_like_version and _version_lt(actual, min_driver):
                raise RuntimeError(
                    f"Brev instance '{instance_name}' nvidia driver is "
                    f"{actual}, task requires at least {min_driver}"
                )
            if not looks_like_version:
                logger.warning(
                    "nvidia-smi on %s returned %r — skipping driver-version "
                    "check (probably no GPU or driver missing).",
                    instance_name, actual,
                )


async def _find_cheapest_matching_type(req: dict) -> str | None:
    """Pick the cheapest `brev search --json` type matching GPU requirements.

    Filters the catalog by gpu_name substring (brev_search or gpu_type),
    gpu_count, per-GPU VRAM, and disk_min_gb floor. Sorts surviving
    candidates by price_per_hour and returns the cheapest type slug
    (e.g. "dmz.h100x2.pcie", "gcpx2-rtx-pro-6000-blackwell-server").
    Returns None if no candidates match — caller raises with the req
    dict so the user sees what to relax.

    Tolerates the older brev CLI variant that prints walkthrough text
    after the JSON array — strip everything past the last `]`.

    Ported from VSS (vss-feat-skill-eval/.github/skill-eval/envs/brev_env.py).
    """
    result = await _run_brev("search", "--json", timeout=30)
    search = (req.get("brev_search") or req.get("gpu_type") or "").lower()
    required_count = int(req.get("gpu_count") or 1)
    required_vram = int(req.get("min_vram_gb_per_gpu") or 0)
    required_disk = int(req.get("min_root_disk_gb") or 0)

    raw = result.stdout or ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        last_bracket = raw.rfind("]")
        if last_bracket < 0:
            return None
        try:
            payload = json.loads(raw[: last_bracket + 1])
        except json.JSONDecodeError:
            return None
    if isinstance(payload, dict):
        payload = payload.get("instances") or []

    candidates = []
    for inst in payload or []:
        if not isinstance(inst, dict):
            continue
        gpu_name = (inst.get("gpu_name") or "").lower()
        gpu_count = int(inst.get("gpu_count") or 0)
        total_vram = float(inst.get("total_vram_gb") or 0)
        disk_min_gb = int(inst.get("disk_min_gb") or 0)
        if search and search not in gpu_name:
            continue
        if gpu_count < required_count:
            continue
        if required_vram and (total_vram / max(gpu_count, 1)) < required_vram:
            continue
        # disk_min_gb is a pre-filter only — some providers (e.g. hyperstack)
        # misreport this; the authoritative check is _check_live_resources
        # after the VM is up.
        if required_disk and disk_min_gb and disk_min_gb < required_disk:
            continue
        candidates.append(inst)
    if not candidates:
        return None
    candidates.sort(key=lambda x: float(x.get("price_per_hour") or 0))
    return candidates[0].get("type")


async def _wait_for_running(name: str, timeout: int = BREV_CREATE_TIMEOUT) -> None:
    """Poll `brev ls` until status=RUNNING AND shell=READY.

    Just checking `status=RUNNING` isn't enough — the VM can flip to RUNNING
    while the SSH daemon / brev shell layer is still initializing, causing
    the first `brev exec` to fail with "command not found" or hang. VSS
    polls both columns; we mirror that.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    last_state = None
    while asyncio.get_event_loop().time() < deadline:
        inst = await _find_brev_instance(name)
        if inst:
            state = (inst.get("status"), inst.get("shell"))
            if state != last_state:
                logger.info("brev %s: status=%s shell=%s", name, *state)
                last_state = state
            if state == ("RUNNING", "READY"):
                return
        await asyncio.sleep(10)
    raise RuntimeError(
        f"Brev instance {name} did not reach RUNNING+READY within {timeout}s "
        f"(last state: {last_state})"
    )

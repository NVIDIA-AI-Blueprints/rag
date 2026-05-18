#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor environment provider for Brev VMs (hybrid mode).

Lifecycle:
    start()    → resolve/create instance, smoke-test exec
    exec()     → brev exec <instance> -- <command>
    upload()   → tar | base64 | brev exec (brev copy has nesting bugs)
    download() → brev exec | base64 -d | tar
    stop()     → brev delete <instance>

Two ways to pick the instance name:

1. Reuse via $BREV_INSTANCE (warm-pool style, e.g. rag-eval-target).
   If the instance exists, use it as-is. If it doesn't, create it.

2. Auto-generate `rag-harbor-<uuid>` when $BREV_INSTANCE unset.
   Ephemeral — always created fresh, always deleted on stop().

Hardware comes from task.toml [metadata]:
    brev_cpu = "4x16"   → brev create --cpu 4x16
    brev_gpu = "..."    → brev create --gpu ...

Adapter (generate.py) writes these from its PLATFORMS dict, keyed off
the user-friendly platform name in the eval spec.
"""
from __future__ import annotations

import asyncio
import base64
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
        # Pick instance name: $BREV_INSTANCE > task.toml override > uuid.
        self._instance_name = (
            DEFAULT_INSTANCE
            or meta.get("brev_instance")
            or f"rag-harbor-{uuid.uuid4().hex[:8]}"
        )
        logger.info("Brev target: %s", self._instance_name)

        existing = await _find_brev_instance(self._instance_name)
        if existing is None:
            await self._provision(meta)
            self._created_by_us = True
        elif existing.get("status") != "RUNNING":
            logger.info("Instance %s exists but status=%s — starting",
                        self._instance_name, existing.get("status"))
            start_result = await _run_brev("start", self._instance_name, timeout=300)
            if start_result.return_code != 0:
                raise RuntimeError(
                    f"brev start {self._instance_name} failed: {start_result.stderr}"
                )
        else:
            logger.info("Reusing existing running instance %s", self._instance_name)

        # Smoke test: confirm we can exec on the instance.
        result = await _run_brev_exec(
            self._instance_name, "echo harbor-ready", timeout=60,
        )
        if result.return_code != 0 or "harbor-ready" not in (result.stdout or ""):
            raise RuntimeError(
                f"Cannot reach Brev instance '{self._instance_name}': "
                f"rc={result.return_code} stderr={result.stderr!r}"
            )

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

        # NOTE: prior-deploy cleanup is NOT done here. Harbor calls start()
        # per trial, but trials within one eval run share a deploy (step-1
        # deploys, step-2 probes). Tearing down between trials breaks that.
        # The script (ci/run_skill_eval.sh) runs the cleanup once, before
        # the harbor loop, so a reused warm-pool VM starts clean.

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

    async def _clean_prior_deploy(self) -> None:
        """Tear down RAG compose stacks on a reused VM, keep image cache.

        Run from start() after _stage_repo. No-op on a fresh VM (no
        containers to remove). Failures are logged but not fatal — the
        agent's own `docker compose up` will surface real conflicts.
        """
        assert self._instance_name
        compose_files = [
            "deploy/compose/docker-compose-rag-server.yaml",
            "deploy/compose/docker-compose-ingestor-server.yaml",
            "deploy/compose/vectordb.yaml",
            "deploy/compose/nims.yaml",
            "deploy/compose/docker-compose-nemo-guardrails.yaml",
            "deploy/compose/observability.yaml",
        ]
        down_cmds = " ; ".join(
            f'[ -f "$HOME/rag/{f}" ] && '
            f'docker compose -f "$HOME/rag/{f}" down -v --remove-orphans '
            f'>/dev/null 2>&1 || true'
            for f in compose_files
        )
        # Also remove the `nvidia-rag` network if it survived (compose down
        # skips it when no compose file references it on this invocation).
        cleanup = (
            f"{down_cmds} ; "
            f"docker network rm nvidia-rag >/dev/null 2>&1 || true ; "
            f"docker ps -a --format '{{{{.Names}}}}' | "
            f"grep -E '(milvus|nv-ingest|rag-server|ingestor|redis|nemo)' | "
            f"xargs -r docker rm -f >/dev/null 2>&1 || true ; "
            f"echo cleanup-done"
        )
        logger.info("Cleaning prior deploy on %s (image cache preserved)",
                    self._instance_name)
        result = await _run_brev_exec(
            self._instance_name, cleanup, timeout=120,
        )
        if result.return_code != 0:
            logger.warning("Prior-deploy cleanup returned rc=%s stderr=%s",
                           result.return_code, result.stderr)

    async def _provision(self, meta: dict) -> None:
        """brev create --type <X> from task.toml metadata.

        Brev CLI v0.6.324+ removed --cpu/--gpu shape flags. Instance type
        names (e.g. `n2d-standard-4`, `g5.xlarge`) are passed via --type.
        Use `brev search cpu` / `brev search gpu` to discover types.
        """
        brev_type = meta.get("brev_type")
        if not brev_type:
            logger.warning(
                "No brev_type in task.toml — falling back to n2d-standard-4"
            )
            brev_type = "n2d-standard-4"

        logger.info("Creating Brev instance %s --type %s",
                    self._instance_name, brev_type)
        create = await _run_brev(
            "create", self._instance_name, "--type", brev_type, "--detached",
            timeout=BREV_CREATE_TIMEOUT,
        )
        if create.return_code != 0:
            raise RuntimeError(
                f"brev create {self._instance_name} --type {brev_type} "
                f"failed: {create.stderr}"
            )
        await _wait_for_running(self._instance_name)

    # -----------------------------------------------------------------------
    # Teardown
    # -----------------------------------------------------------------------

    async def stop(self, delete: bool) -> None:
        if not self._instance_name:
            return
        # Named-pool mode ($BREV_INSTANCE set): NEVER delete here, even
        # when Harbor passes delete=True between trials. The named VM
        # must survive across step-N invocations so step-2 can probe the
        # RAG deployment that step-1 brought up. The caller
        # (ci/run_skill_eval.sh) explicitly deletes after all trials.
        if DEFAULT_INSTANCE:
            logger.info("Named pool %s — stop() is a no-op (script "
                        "handles cleanup after all trials)",
                        self._instance_name)
            return
        # Ephemeral mode (rag-harbor-<uuid>): each trial owns its own VM
        # and should clean up. Honour Harbor's delete flag.
        if self._created_by_us or delete:
            logger.info("Deleting Brev instance %s", self._instance_name)
            await _run_brev("delete", self._instance_name, timeout=120)

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


async def _run_brev(*args: str, timeout: int = 30) -> ExecResult:
    """Invoke `brev <args>` as a subprocess and return its result.

    Stdin is closed via DEVNULL. `brev create` reads piped stdin as a
    fallback list of instance types (`brev search | brev create ...`).
    Without DEVNULL, brev inherits whatever's on our stdin — and our
    caller (ci/run_skill_eval.sh) runs harbor inside a `while read ...
    done < <(find ...)` loop, leaving the rest of the find output on
    stdin. brev parses that as bogus "next type" fallbacks and the
    create command fails on transient API errors instead of retrying.
    """
    proc = await asyncio.create_subprocess_exec(
        "brev", *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
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

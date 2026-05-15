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
        await _run_brev_exec(
            self._instance_name,
            "sudo mkdir -p /logs /tests /solution /skills /installed-agent && "
            'sudo chown -R "$(whoami):$(id -gn)" '
            "/logs /tests /solution /skills /installed-agent",
            timeout=60,
        )
        self._started = True

    async def _provision(self, meta: dict) -> None:
        """brev create with --cpu/--gpu flags from task.toml metadata."""
        flags: list[str] = []
        if meta.get("brev_cpu"):
            flags = ["--cpu", str(meta["brev_cpu"])]
        elif meta.get("brev_gpu"):
            flags = ["--gpu", str(meta["brev_gpu"])]
        else:
            # Fallback for tasks without resource metadata.
            logger.warning("No brev_cpu / brev_gpu in task.toml — using --cpu 4x16")
            flags = ["--cpu", "4x16"]

        logger.info("Creating Brev instance %s %s", self._instance_name, flags)
        create = await _run_brev(
            "create", self._instance_name, "--detached", *flags,
            timeout=BREV_CREATE_TIMEOUT,
        )
        if create.return_code != 0:
            raise RuntimeError(
                f"brev create {self._instance_name} failed: {create.stderr}"
            )
        await _wait_for_running(self._instance_name)

    # -----------------------------------------------------------------------
    # Teardown
    # -----------------------------------------------------------------------

    async def stop(self, delete: bool) -> None:
        if not self._instance_name:
            return
        # Always delete what we created. For reused warm-pool instances
        # ($BREV_INSTANCE pointing to an existing VM), only delete if the
        # caller asked for it.
        should_delete = self._created_by_us or delete
        if should_delete:
            logger.info("Deleting Brev instance %s", self._instance_name)
            await _run_brev("delete", self._instance_name, timeout=120)
        else:
            logger.info("Leaving Brev instance %s running (reused, not ours)",
                        self._instance_name)

    # -----------------------------------------------------------------------
    # File transfer
    # -----------------------------------------------------------------------

    async def upload_file(
        self, source_path: Path | str, target_path: str,
    ) -> None:
        assert self._instance_name
        src = Path(source_path)
        # Encode locally, decode on the remote — `brev copy` has flaky
        # behavior for nested files / special chars.
        encoded = base64.b64encode(src.read_bytes()).decode()
        target_dir = str(Path(target_path).parent) or "/"
        result = await _run_brev_exec(
            self._instance_name,
            f"sudo mkdir -p {shlex.quote(target_dir)} && "
            f'sudo chown "$(whoami):$(id -gn)" {shlex.quote(target_dir)} && '
            f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(target_path)}",
            timeout=BREV_COPY_TIMEOUT,
        )
        if result.return_code != 0:
            raise RuntimeError(f"upload_file failed: {result.stderr}")

    async def upload_dir(
        self, source_dir: Path | str, target_dir: str,
    ) -> None:
        assert self._instance_name
        src = str(source_dir).rstrip("/")
        tar_bytes = subprocess.check_output(
            ["tar", "-czf", "-", "-C", src, "."], timeout=120,
        )
        encoded = base64.b64encode(tar_bytes).decode()
        result = await _run_brev_exec(
            self._instance_name,
            f"sudo mkdir -p {shlex.quote(target_dir)} && "
            f'sudo chown "$(whoami):$(id -gn)" {shlex.quote(target_dir)} && '
            f"echo {shlex.quote(encoded)} | base64 -d | "
            f"tar -xzf - -C {shlex.quote(target_dir)}",
            timeout=BREV_COPY_TIMEOUT,
        )
        if result.return_code != 0:
            raise RuntimeError(f"upload_dir failed: {result.stderr}")

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
    """Invoke `brev <args>` as a subprocess and return its result."""
    proc = await asyncio.create_subprocess_exec(
        "brev", *args,
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
    return await _run_brev("exec", instance, "--", "bash", "-c", command, timeout=timeout)


async def _find_brev_instance(name: str) -> dict | None:
    """Return the instance row from `brev ls` or None if not found."""
    result = await _run_brev("ls", timeout=30)
    if result.return_code != 0:
        return None
    # `brev ls` is human-formatted, not JSON. Parse table rows.
    for line in (result.stdout or "").splitlines():
        cols = line.split()
        if len(cols) >= 2 and cols[0] == name:
            return {"name": cols[0], "status": cols[1]}
    return None


async def _wait_for_running(name: str, timeout: int = BREV_CREATE_TIMEOUT) -> None:
    """Poll `brev ls` until the named instance is RUNNING (or timeout)."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        inst = await _find_brev_instance(name)
        if inst and inst.get("status") == "RUNNING":
            return
        await asyncio.sleep(10)
    raise RuntimeError(f"Brev instance {name} did not reach RUNNING within {timeout}s")

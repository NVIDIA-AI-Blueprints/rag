#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LocalEnvironment — run Harbor trials directly on the host (no Docker/Brev).

The ClaudeCode agent hard-codes several paths that normally live inside a
container:
    CLAUDE_CONFIG_DIR = /logs/agent/sessions
    tee /logs/agent/claude-code.txt
    /logs/verifier/reward.txt

This environment maps those root-level paths into a per-session workdir
under <skill-eval>/harbor-workdir/<session_id>/ via regex substitution so
the agent never needs root access.
"""
from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths


# Anchor everything at the skill-eval/ directory containing this file
# (envs/local_env.py → parent.parent = skill-eval/). Lets the same env
# work for any host/checkout without editing a hardcoded path.
_SKILL_EVAL_DIR = Path(__file__).resolve().parent.parent

# Root-level paths the Harbor agent/verifier reference inside the "container".
_MAPPED_PREFIXES: list[str] = ["/logs/", "/tests/", "/solution/", "/skills/", "/installed-agent"]

# Command fragments that are no-ops on a local host where everything is pre-installed.
_SKIP_FRAGMENTS: tuple[str, ...] = (
    "apt-get",
    "apk add",
    "yum install",
    "npm install -g @anthropic-ai/claude-code",
    "curl -fsSL https://claude.ai/install.sh",
    "claude.ai/install.sh",
)


class LocalEnvironment(BaseEnvironment):
    """Run Harbor trials directly on the localhost — no container, no GPU."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _base = _SKILL_EVAL_DIR / "harbor-workdir"
        self._workdir = _base / self.session_id
        try:
            self._workdir.mkdir(parents=True, exist_ok=True)
            for subdir in ("logs/agent/sessions", "logs/verifier", "logs/artifacts",
                           "tests", "solution", "skills"):
                (self._workdir / subdir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            _base.mkdir(parents=True, exist_ok=True)
            (_base / "mkdir_error.debug").write_text(
                f"session={self.session_id}\nworkdir={self._workdir}\nerror={exc}\n"
            )

    def _map_path(self, path: str) -> str:
        for prefix in _MAPPED_PREFIXES:
            if path.startswith(prefix):
                return str(self._workdir / path.lstrip("/"))
            if path == prefix.rstrip("/"):
                return str(self._workdir / path.lstrip("/"))
        return path

    def _map_command(self, command: str) -> str:
        """Replace hard-coded Harbor root-level paths in a shell command string.

        Uses regex with a path-boundary lookbehind so we only rewrite a
        prefix like ``/logs/`` when it starts a path (preceded by a non-path
        char like space, quote, parenthesis, or start of string), NOT when
        it's a coincidental substring inside an already-absolute path like
        ``/home/.../harbor-workdir/<sess>/logs/...``.
        """
        import re
        result = command
        for prefix in _MAPPED_PREFIXES:
            bare = prefix.rstrip("/")
            replacement_dir = str(self._workdir / bare.lstrip("/"))
            pattern = r"(?:(?<=[\s\"'(>;|&=`])|(?<=^))" + re.escape(prefix)
            result = re.sub(pattern, replacement_dir + "/", result)
            pattern_bare = (
                r"(?:(?<=[\s\"'(>;|&=`])|(?<=^))"
                + re.escape(bare)
                + r"(?=[\s\"')>;|&]|$)"
            )
            result = re.sub(pattern_bare, replacement_dir, result)
        return result

    @staticmethod
    def type() -> str:
        return "local"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(gpus=False, disable_internet=False, mounted=False)

    @property
    def env_paths(self) -> EnvironmentPaths:
        root = self._workdir
        logs_dir = root / "logs"
        verifier_dir = logs_dir / "verifier"
        from pathlib import PurePosixPath
        return EnvironmentPaths(
            logs_dir=PurePosixPath(str(logs_dir)),
            agent_dir=PurePosixPath(str(logs_dir / "agent")),
            verifier_dir=PurePosixPath(str(verifier_dir)),
            artifacts_dir=PurePosixPath(str(logs_dir / "artifacts")),
            tests_dir=PurePosixPath(str(root / "tests")),
            solution_dir=PurePosixPath(str(root / "solution")),
            reward_text_path=PurePosixPath(str(verifier_dir / "reward.txt")),
            reward_json_path=PurePosixPath(str(verifier_dir / "reward.json")),
        )

    def _validate_definition(self) -> None:
        pass

    async def start(self, force_build: bool = False) -> None:
        pass

    async def stop(self, delete: bool = False) -> None:
        pass

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        bare = command.replace("set -o pipefail; ", "").strip()
        if any(frag in bare for frag in _SKIP_FRAGMENTS):
            return ExecResult(stdout="skipped (already installed on host)", stderr="", return_code=0)

        mapped_cmd = self._map_command(command)
        merged_env = self._merge_env(env)
        proc_env = {**os.environ}
        if merged_env:
            proc_env.update(merged_env)
        # generic_judge.py reads DFW_VERIFIER_DIR to know where to write reward.txt
        # (kept the env-var name for code reuse with the DFW skill-eval).
        proc_env["DFW_VERIFIER_DIR"] = str(self._workdir / "logs" / "verifier")
        # generic_judge.py's _TRAJECTORY_CANDIDATES are hard-coded to /logs/...
        # which only exist inside Harbor containers. On LocalEnvironment the
        # trajectory lives in the workdir; expose the actual path so the
        # judge can find it.
        proc_env["DFW_TRAJECTORY_FILE"] = str(
            self._workdir / "logs" / "agent" / "claude-code.txt"
        )

        # CLAUDE_CONFIG_DIR handling — split by command type:
        #   * claude --verbose / claude --print (the agent invocation):
        #     UNSET it. Setting CLAUDE_CONFIG_DIR (even to its default
        #     ~/.claude) makes claude skip SSE-channel auth from the
        #     parent Claude Code session and look for OAuth on disk —
        #     which on SSO-managed installs isn't persisted. Unset →
        #     claude uses its default + SSE auth → works.
        #   * setup commands (`mkdir -p $CLAUDE_CONFIG_DIR/debug ...`,
        #     skills copy): set it to a writable per-trial workdir so
        #     the mkdir/cp commands succeed. These commands don't need
        #     auth and the per-trial workdir is fine for them.
        is_claude_invocation = (
            "claude --verbose" in command or "claude --print" in command
        )
        if is_claude_invocation:
            proc_env.pop("CLAUDE_CONFIG_DIR", None)
        else:
            proc_env["CLAUDE_CONFIG_DIR"] = str(
                self._workdir / "logs" / "agent" / "sessions"
            )

        # Do NOT pop ANTHROPIC_API_KEY here. When the parent shell is a
        # Claude Code session, ANTHROPIC_API_KEY is set to "" (empty string)
        # and CLAUDE_CODE_SSE_PORT is set — together those tell the spawned
        # `claude` CLI to authenticate via the SSE channel back to the live
        # CC session. Removing ANTHROPIC_API_KEY entirely makes claude treat
        # the subprocess as standalone and look for OAuth on disk, which on
        # SSO-managed (NVIDIA enterprise) installs is not persisted to a
        # readable file → "Not logged in · Please run /login".
        # When ANTHROPIC_BASE_URL is set (e.g. inference-api.nvidia.com proxy
        # path), the existing ANTHROPIC_API_KEY value (a real sk-* key) is
        # what claude uses; SSE auth is bypassed in that case.

        _NO_REMAP = {"CLAUDE_CONFIG_DIR"}
        for k, v in list(proc_env.items()):
            if k in _NO_REMAP:
                continue
            if isinstance(v, str):
                mapped_v = self._map_command(v)
                if mapped_v != v:
                    proc_env[k] = mapped_v

        try:
            keys_of_interest = [k for k in proc_env if any(s in k for s in ('ANTHROPIC', 'JUDGE', 'CLAUDE'))]
            env_dump = "\n".join(f"  {k}={proc_env[k]!r}" for k in sorted(keys_of_interest))
            (self._workdir / "exec.debug.log").open("a").write(
                f"--- exec ---\ncmd: {mapped_cmd}\ncwd: {cwd}\nenv-values:\n{env_dump}\n"
            )
        except Exception:
            pass
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash", "-c", mapped_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=proc_env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=float(timeout_sec) if timeout_sec else None,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return ExecResult(
                stdout="",
                stderr=f"Command timed out after {timeout_sec}s",
                return_code=124,
            )

        stdout_str = stdout_b.decode("utf-8", errors="replace")
        stderr_str = stderr_b.decode("utf-8", errors="replace")
        try:
            (self._workdir / "exec.debug.log").open("a").write(
                f"rc: {proc.returncode}\n"
                f"stdout[-500:]: {stdout_str[-500:]!r}\n"
                f"stderr[-500:]: {stderr_str[-500:]!r}\n\n"
            )
        except Exception:
            pass
        return ExecResult(
            stdout=stdout_str,
            stderr=stderr_str,
            return_code=proc.returncode or 0,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        target = Path(self._map_path(target_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source_path), str(target))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        target = Path(self._map_path(target_dir))
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(source_dir), str(target), dirs_exist_ok=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        source = Path(self._map_path(source_path))
        target = Path(str(target_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        source = Path(self._map_path(source_dir))
        target = Path(str(target_dir))
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(source), str(target), dirs_exist_ok=True)

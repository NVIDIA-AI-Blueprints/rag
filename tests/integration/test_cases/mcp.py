"""
MCP end-to-end integration sequence
Numbered steps:
  86) Create collection for MCP
  87) Upload files to a collection (ingestor API)
  88) Start MCP server (SSE) in background
  89) SSE: List tools
  90) SSE: Call generate
  91) SSE: Call search
  92) SSE: Call get_summary
  93) stdio: List tools
  94) stdio: Call generate
  95) stdio: Call search
  96) stdio: Call get_summary
  97) Start MCP server (streamable_http)
  98) streamable_http: List tools
  99) streamable_http: Call generate
 100) streamable_http: Call search
 101) streamable_http: Call get_summary
 102) MCP: Delete test collection and stop MCP servers
"""

import json
import asyncio
import logging
import os
import shlex
import subprocess
import sys
import time
from typing import Any

import aiohttp
from urllib.parse import urlparse

from ..base import BaseTestModule, TestStatus, test_case

logger = logging.getLogger(__name__)

try:
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    try:
        from mcp.client.stdio import StdioServerParameters  # type: ignore
    except Exception:
        StdioServerParameters = None  # type: ignore
    from mcp.client.session import ClientSession
except Exception:
    # The test runner should skip this module if MCP SDK isn't available
    pass


class MCPIntegrationModule(BaseTestModule):
    """End-to-end MCP integration module (script-like, numbered steps)."""

    def __init__(self, test_runner):
        super().__init__(test_runner)
        self.collection = "test"
        # Server exposes SSE at /sse/ and message endpoint at /messages/
        # Prefer the SSE path as base URL for the SSE client.
        self.sse_url = "http://127.0.0.1:8000/sse"
        # Prefer NVCF if present
        self.api_key = os.getenv("NVCF_API_KEY") or os.getenv("NVIDIA_API_KEY") or ""
        self.sse_proc: subprocess.Popen | None = None
        self.stream_proc: subprocess.Popen | None = None

    def _headers(self) -> dict[str, str]:
        hdrs: dict[str, str] = {}
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        return hdrs

    async def _upload_files(self, files: list[str]) -> bool:
        """Upload small set of files to ingestor to prepare the collection."""
        if not files:
            logger.warning("No files to upload for MCP tests, continuing without KB.")
            return True
        data = {
            "collection_name": self.collection,
            "blocking": True,
            "split_options": {"chunk_size": 512, "chunk_overlap": 150},
            "custom_metadata": [],
            "generate_summary": True,
        }
        form_data = aiohttp.FormData()
        for file_path in files:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    form_data.add_field(
                        "documents",
                        f.read(),
                        filename=os.path.basename(file_path),
                        content_type="application/octet-stream",
                    )
        form_data.add_field("data", json.dumps(data), content_type="application/json")

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.ingestor_server_url}/v1/documents", data=form_data) as resp:
                ok = resp.status == 200
                try:
                    result = await resp.json()
                    logger.info("Upload response: %s", json.dumps(result, indent=2))
                except Exception:
                    logger.info("Upload response text: %s", await resp.text())
                return ok

    def _start_sse_server(self) -> None:
        env = dict(os.environ)
        if self.api_key:
            # Provide both env and --api-key to be robust
            env.setdefault("NVCF_API_KEY", self.api_key)
            env.setdefault("NVIDIA_API_KEY", self.api_key)
        # Best-effort free the port before starting
        try:
            self._free_server_port()
        except Exception:
            pass
        cmd = [
            sys.executable,
            "-m",
            "nvidia_rag.utils.mcp.mcp_server",
            "--transport",
            "sse",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
        if self.api_key:
            cmd += ["--api-key", self.api_key]
        logger.info("Launching SSE MCP server: %s", " ".join(shlex.quote(c) for c in cmd))
        # Suppress server stdout/stderr to minimize noisy logs in test output
        self.sse_proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("SSE server PID: %s", getattr(self.sse_proc, "pid", None))

    def _start_streamable_http_server(self) -> None:
        """Start the MCP server in streamable_http mode (background)."""
        env = dict(os.environ)
        if self.api_key:
            env.setdefault("NVCF_API_KEY", self.api_key)
            env.setdefault("NVIDIA_API_KEY", self.api_key)
        # Best-effort free the shared HTTP port before starting streamable_http server
        try:
            self._free_server_port()
        except Exception:
            pass
        cmd = [
            sys.executable,
            "-m",
            "nvidia_rag.utils.mcp.mcp_server",
            "--transport",
            "streamable_http",
        ]
        if self.api_key:
            cmd += ["--api-key", self.api_key]
        logger.info("Launching streamable_http MCP server: %s", " ".join(shlex.quote(c) for c in cmd))
        # Suppress server stdout/stderr to minimize noisy logs in test output
        self.stream_proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("streamable_http server PID: %s", getattr(self.stream_proc, "pid", None))

    def _stop_server(self, timeout: float = 5.0) -> bool:
        """Stop the HTTP MCP server subprocess if running."""
        proc = getattr(self, "sse_proc", None)
        if not proc:
            # Even if we don't have a handle, free the port in case of orphan
            try:
                self._free_server_port()
            except Exception:
                pass
            return True
        try:
            if proc.poll() is None:
                proc.terminate()
                start = time.time()
                while proc.poll() is None and time.time() - start < timeout:
                    time.sleep(0.1)
            if proc.poll() is None:
                proc.kill()
            # After stopping, free the port in case child processes linger
            try:
                self._free_server_port()
            except Exception:
                pass
            return True
        except Exception:
            return False
        finally:
            self.sse_proc = None

    async def _wait_for_sse_ready(self, timeout: float = 20.0, interval: float = 0.5) -> bool:
        """
        Poll SSE endpoints until the MCP server is ready or timeout occurs.
        Tries /sse and /messages paths to accommodate different SDK versions.
        """
        start = time.time()
        base = (self.sse_url or "").rstrip("/")
        root = base[:-4] if base.endswith("/sse") else base
        candidates = [
            base,
            base + "/",
            root + "/sse",
            root + "/sse/",
            root + "/messages",
            root + "/messages/",
        ]
        headers = self._headers()
        last_error = None
        while time.time() - start < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    for url in candidates:
                        try:
                            async with session.get(url, headers=headers, timeout=5) as resp:
                                if 200 <= resp.status < 300:
                                    return True
                                last_error = f"{url} -> {resp.status}"
                        except Exception as e:
                            last_error = f"{url} -> {e}"
            except Exception as e:
                last_error = str(e)
            await asyncio.sleep(interval)
        logger.warning("Timed out waiting for SSE server readiness: %s", last_error)
        return False

    async def _sse_session(self):
        return sse_client(url=self.sse_url, headers=self._headers())

    def _dict_contains_keys_ci(self, obj, keys: tuple[str, ...]) -> bool:
        """Recursively check if a dict/list structure contains any of the keys (case-insensitive)."""
        try:
            if isinstance(obj, dict):
                lower_keys = {str(k).lower() for k in obj.keys()}
                if any(k in lower_keys for k in keys):
                    return True
                for v in obj.values():
                    if self._dict_contains_keys_ci(v, keys):
                        return True
            elif isinstance(obj, list):
                for item in obj:
                    if self._dict_contains_keys_ci(item, keys):
                        return True
        except Exception:
            pass
        return False

    def _get_sse_port(self) -> int:
        """Extract the TCP port from the SSE URL; defaults to 8000."""
        base = self._sse_base_url()
        try:
            parsed = urlparse(base)
            if parsed.port:
                return int(parsed.port)
        except Exception:
            pass
        return 8000

    def _free_server_port(self) -> None:
        """Attempt to kill any process listening on the shared HTTP MCP port."""
        port = self._get_sse_port()
        # Try fuser
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False, capture_output=True, text=True)
        except Exception:
            pass
        # Try lsof as a fallback
        try:
            out = subprocess.run(["lsof", "-ti", f"tcp:{port}"], check=False, capture_output=True, text=True)
            pids = [p.strip() for p in out.stdout.splitlines() if p.strip().isdigit()]
            for pid in pids:
                try:
                    os.kill(int(pid), 15)
                except Exception:
                    pass
        except Exception:
            pass

    def _sse_base_url(self) -> str:
        """Return base URL suitable for mcp_client SSE (without trailing /sse)."""
        url = (self.sse_url or "").rstrip("/")
        return url[:-4] if url.endswith("/sse") else url

    def _stream_base_url(self) -> str:
        """Base URL for streamable_http MCP client calls."""
        return "http://127.0.0.1:8000"

    def _mcp_client_cmd(self) -> list[str]:
        return [sys.executable, "-m", "nvidia_rag.utils.mcp.mcp_client"]

    def _run_mcp_client(self, args: list[str], extra_env: dict[str, str] | None = None, timeout: float = 60.0) -> tuple[int, str, str]:
        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)
        proc = subprocess.run(self._mcp_client_cmd() + args, capture_output=True, text=True, env=env, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr

    async def _stdio_session(self):
        cmd = sys.executable
        args = ["-m", "nvidia_rag.utils.mcp.mcp_server", "--transport", "stdio"]
        if self.api_key:
            args += ["--api-key", self.api_key]
        if StdioServerParameters is not None:
            params = StdioServerParameters(command=cmd, args=args)  # type: ignore[call-arg]
            return stdio_client(params)
        # Fall back across possible legacy signatures
        try:
            return stdio_client(command=cmd, args=args)  # type: ignore[misc]
        except TypeError:
            pass
        try:
            return stdio_client(cmd, args)  # type: ignore[misc]
        except TypeError:
            pass
        try:
            return stdio_client([cmd] + args)  # type: ignore[misc]
        except TypeError:
            pass
        return stdio_client(cmd, " ".join(args))  # type: ignore[misc]

    @test_case(86, "Create MCP Collection")
    async def create_mcp_collection(self) -> bool:
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ingestor_server_url}/v1/collections",
                    json=[self.collection],
                ) as resp:
                    ok = resp.status in (200, 201)
        except Exception:
            ok = False
        self.add_test_result(
            86,
            "Create MCP Collection",
            f"Create test collection '{self.collection}' for MCP flows.",
            ["POST /v1/collections"],
            ["collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "Failed to create MCP test collection",
        )
        return ok

    @test_case(87, "Upload Test Files for MCP")
    async def upload_test_files_for_mcp(self) -> bool:
        start = time.time()
        # Reuse a small default file from data dir if available
        default_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "..",
            "..",
            "data",
            "multimodal",
            "woods_frost.pdf",
        )
        files = [default_file] if os.path.exists(default_file) else []
        ok = await self._upload_files(files)
        self.add_test_result(
            87,
            "Upload Test Files for MCP",
            f"Upload sample file(s) to collection '{self.collection}' to enable search/summary.",
            ["POST /v1/documents"],
            ["collection_name", "blocking", "generate_summary"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "Upload failed",
        )
        return ok

    @test_case(88, "Start MCP Server (SSE)")
    async def start_mcp_server_sse(self) -> bool:
        start = time.time()
        try:
            self._start_sse_server()
            ready = await self._wait_for_sse_ready(timeout=30.0, interval=0.5)
            status = TestStatus.SUCCESS if ready else TestStatus.FAILURE
            err = None if ready else "SSE MCP server did not become ready in time"
        except Exception as e:
            status = TestStatus.FAILURE
            err = str(e)
        self.add_test_result(
            88,
            "Start MCP Server (SSE)",
            "Launch MCP server over SSE on http://127.0.0.1:8000.",
            ["MCP/SSE server"],
            [],
            time.time() - start,
            status,
            None if status == TestStatus.SUCCESS else "SSE MCP server did not become ready in time",
        )
        return status == TestStatus.SUCCESS

    @test_case(89, "SSE: List Tools")
    async def sse_list_tools(self) -> bool:
        start = time.time()
        try:
            args = ["list", "--transport", "sse", "--url", self._sse_base_url()]
            if self.api_key:
                args += ["--header", f"Authorization: Bearer {self.api_key}"]
            code, out, err_str = self._run_mcp_client(args)
            logger.info("MCP client output (SSE list): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
            err = None if ok else "SSE list tools failed"
        except Exception as e:
            ok, err = False, str(e)
        self.add_test_result(
            89,
            "SSE: List Tools",
            "List available MCP tools over SSE.",
            ["MCP/SSE list_tools"],
            [],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "SSE list tools did not include all required tools",
        )
        return ok

    @test_case(90, "SSE: Call Generate")
    async def sse_call_generate(self) -> bool:
        start = time.time()
        try:
            payload = {
                "messages": [{"role": "user", "content": "Say 'ok'"}],
                "collection_name": self.collection,
            }
            args = [
                "call",
                "--transport",
                "sse",
                "--url",
                self._sse_base_url(),
                "--tool",
                "generate",
                "--json-args",
                json.dumps(payload),
            ]
            if self.api_key:
                args += ["--header", f"Authorization: Bearer {self.api_key}"]
            code, out, err_str = self._run_mcp_client(args)
            logger.info("MCP client output (SSE generate): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
            err = None if ok else "SSE generate failed"
        except Exception as e:
            ok, err = False, str(e)
        self.add_test_result(
            90,
            "SSE: Call Generate",
            "Call 'generate' tool over SSE.",
            ["MCP/SSE call_tool(generate)"],
            ["messages", "collection_name"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "SSE generate did not return expected content",
        )
        return ok

    @test_case(91, "SSE: Call Search")
    async def sse_call_search(self) -> bool:
        start = time.time()
        try:
            payload = {
                "query": "woods frost",
                "collection_name": self.collection,
            }
            args = [
                "call",
                "--transport",
                "sse",
                "--url",
                self._sse_base_url(),
                "--tool",
                "search",
                "--json-args",
                json.dumps(payload),
            ]
            if self.api_key:
                args += ["--header", f"Authorization: Bearer {self.api_key}"]
            code, out, err_str = self._run_mcp_client(args)
            logger.info("MCP client output (SSE search): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
            err = None if ok else "SSE search failed"
        except Exception as e:
            ok, err = False, str(e)
        self.add_test_result(
            91,
            "SSE: Call Search",
            "Call 'search' tool over SSE.",
            ["MCP/SSE call_tool(search)"],
            ["query", "collection_name"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "SSE search did not return results",
        )
        return ok

    @test_case(92, "SSE: Call Get Summary")
    async def sse_call_get_summary(self) -> bool:
        start = time.time()
        try:
            payload = {
                "collection_name": self.collection,
                "file_name": "woods_frost.pdf",
                "blocking": False,
                "timeout": 60,
            }
            args = [
                "call",
                "--transport",
                "sse",
                "--url",
                self._sse_base_url(),
                "--tool",
                "get_summary",
                "--json-args",
                json.dumps(payload),
            ]
            if self.api_key:
                args += ["--header", f"Authorization: Bearer {self.api_key}"]
            code, out, err_str = self._run_mcp_client(args)
            logger.info("MCP client output (SSE get_summary): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
            err = None if ok else "SSE get_summary failed"
        except Exception as e:
            ok, err = False, str(e)
        self.add_test_result(
            92,
            "SSE: Call Get Summary",
            "Call 'get_summary' tool over SSE.",
            ["MCP/SSE call_tool(get_summary)"],
            ["collection_name", "file_name", "blocking", "timeout"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "SSE get_summary did not return expected fields",
        )
        return ok

    @test_case(98, "streamable_http: List Tools")
    async def streamable_http_list_tools(self) -> bool:
        start = time.time()
        try:
            args = [
                "list",
                "--transport",
                "streamable_http",
                "--url",
                self._stream_base_url(),
            ]
            if self.api_key:
                args += ["--header", f"Authorization: Bearer {self.api_key}"]
            code, out, err_str = self._run_mcp_client(args)
            logger.info("MCP client output (streamable_http list): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
        except Exception as e:
            ok = False
        self.add_test_result(
            98,
            "streamable_http: List Tools",
            "List available MCP tools over streamable_http.",
            ["MCP/streamable_http list_tools"],
            [],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http list tools failed",
        )
        return ok

    @test_case(99, "streamable_http: Call Generate")
    async def streamable_http_call_generate(self) -> bool:
        start = time.time()
        try:
            payload = {
                "messages": [{"role": "user", "content": "Say 'ok'"}],
                "collection_name": self.collection,
            }
            args = [
                "call",
                "--transport",
                "streamable_http",
                "--url",
                self._stream_base_url(),
                "--tool",
                "generate",
                "--json-args",
                json.dumps(payload),
            ]
            if self.api_key:
                args += ["--header", f"Authorization: Bearer {self.api_key}"]
            code, out, err_str = self._run_mcp_client(args)
            logger.info("MCP client output (streamable_http generate): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
        except Exception as e:
            ok = False
        self.add_test_result(
            99,
            "streamable_http: Call Generate",
            "Call 'generate' tool over streamable_http.",
            ["MCP/streamable_http call_tool(generate)"],
            ["messages", "collection_name"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http generate failed",
        )
        return ok

    @test_case(100, "streamable_http: Call Search")
    async def streamable_http_call_search(self) -> bool:
        start = time.time()
        try:
            payload = {
                "query": "woods frost",
                "collection_name": self.collection,
            }
            args = [
                "call",
                "--transport",
                "streamable_http",
                "--url",
                self._stream_base_url(),
                "--tool",
                "search",
                "--json-args",
                json.dumps(payload),
            ]
            if self.api_key:
                args += ["--header", f"Authorization: Bearer {self.api_key}"]
            code, out, err_str = self._run_mcp_client(args)
            logger.info("MCP client output (streamable_http search): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
        except Exception as e:
            ok = False
        self.add_test_result(
            100,
            "streamable_http: Call Search",
            "Call 'search' tool over streamable_http.",
            ["MCP/streamable_http call_tool(search)"],
            ["query", "collection_name"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http search failed",
        )
        return ok

    @test_case(101, "streamable_http: Call Get Summary")
    async def streamable_http_call_get_summary(self) -> bool:
        start = time.time()
        try:
            payload = {
                "collection_name": self.collection,
                "file_name": "woods_frost.pdf",
                "blocking": False,
                "timeout": 60,
            }
            args = [
                "call",
                "--transport",
                "streamable_http",
                "--url",
                self._stream_base_url(),
                "--tool",
                "get_summary",
                "--json-args",
                json.dumps(payload),
            ]
            if self.api_key:
                args += ["--header", f"Authorization: Bearer {self.api_key}"]
            code, out, err_str = self._run_mcp_client(args)
            logger.info("MCP client output (streamable_http get_summary): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
        except Exception as e:
            ok = False
        self.add_test_result(
            101,
            "streamable_http: Call Get Summary",
            "Call 'get_summary' tool over streamable_http.",
            ["MCP/streamable_http call_tool(get_summary)"],
            ["collection_name", "file_name", "blocking", "timeout"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http get_summary failed",
        )
        return ok

    @test_case(93, "stdio: List Tools")
    async def stdio_list_tools(self) -> bool:
        start = time.time()
        try:
            args = [
                "list",
                "--transport",
                "stdio",
                "--command",
                sys.executable,
                "--args",
                "-m nvidia_rag.utils.mcp.mcp_server --transport stdio" + (f" --api-key {self.api_key}" if self.api_key else ""),
            ]
            # Also pass env for robustness
            extra_env = {}
            if self.api_key:
                extra_env["NVIDIA_API_KEY"] = self.api_key
            code, out, err_str = self._run_mcp_client(args, extra_env=extra_env)
            logger.info("MCP client output (stdio list): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
            err = None if ok else "stdio list tools failed"
        except Exception as e:
            ok, err = False, str(e)
        self.add_test_result(
            93,
            "stdio: List Tools",
            "List available MCP tools over stdio.",
            ["MCP/stdio list_tools"],
            [],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "stdio list tools did not include all required tools",
        )
        return ok

    @test_case(94, "stdio: Call Generate")
    async def stdio_call_generate(self) -> bool:
        start = time.time()
        try:
            payload = {
                "messages": [{"role": "user", "content": "Say 'ok'"}],
                "collection_name": self.collection,
            }
            args = [
                "call",
                "--transport",
                "stdio",
                "--command",
                sys.executable,
                "--args",
                "-m nvidia_rag.utils.mcp.mcp_server --transport stdio" + (f" --api-key {self.api_key}" if self.api_key else ""),
                "--tool",
                "generate",
                "--json-args",
                json.dumps(payload),
            ]
            extra_env = {}
            if self.api_key:
                extra_env["NVIDIA_API_KEY"] = self.api_key
            code, out, err_str = self._run_mcp_client(args, extra_env=extra_env)
            logger.info("MCP client output (stdio generate): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
            err = None if ok else "stdio generate failed"
        except Exception as e:
            ok, err = False, str(e)
        self.add_test_result(
            94,
            "stdio: Call Generate",
            "Call 'generate' tool over stdio.",
            ["MCP/stdio call_tool(generate)"],
            ["messages", "collection_name"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "stdio generate did not return expected content",
        )
        return ok

    @test_case(95, "stdio: Call Search")
    async def stdio_call_search(self) -> bool:
        start = time.time()
        try:
            payload = {
                "query": "woods frost",
                "collection_name": self.collection,
            }
            args = [
                "call",
                "--transport",
                "stdio",
                "--command",
                sys.executable,
                "--args",
                "-m nvidia_rag.utils.mcp.mcp_server --transport stdio" + (f" --api-key {self.api_key}" if self.api_key else ""),
                "--tool",
                "search",
                "--json-args",
                json.dumps(payload),
            ]
            extra_env = {}
            if self.api_key:
                extra_env["NVIDIA_API_KEY"] = self.api_key
            code, out, err_str = self._run_mcp_client(args, extra_env=extra_env)
            logger.info("MCP client output (stdio search): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
            err = None if ok else "stdio search failed"
        except Exception as e:
            ok, err = False, str(e)
        self.add_test_result(
            95,
            "stdio: Call Search",
            "Call 'search' tool over stdio.",
            ["MCP/stdio call_tool(search)"],
            ["query", "collection_name"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "stdio search did not return results",
        )
        return ok

    @test_case(96, "stdio: Call Get Summary")
    async def stdio_call_get_summary(self) -> bool:
        start = time.time()
        try:
            payload = {
                "collection_name": self.collection,
                "file_name": "woods_frost.pdf",
                "blocking": False,
                "timeout": 60,
            }
            args = [
                "call",
                "--transport",
                "stdio",
                "--command",
                sys.executable,
                "--args",
                "-m nvidia_rag.utils.mcp.mcp_server --transport stdio" + (f" --api-key {self.api_key}" if self.api_key else ""),
                "--tool",
                "get_summary",
                "--json-args",
                json.dumps(payload),
            ]
            extra_env = {}
            if self.api_key:
                extra_env["NVIDIA_API_KEY"] = self.api_key
            code, out, err_str = self._run_mcp_client(args, extra_env=extra_env)
            logger.info("MCP client output (stdio get_summary): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
            err = None if ok else "stdio get_summary failed"
        except Exception as e:
            ok, err = False, str(e)
        self.add_test_result(
            96,
            "stdio: Call Get Summary",
            "Call 'get_summary' tool over stdio.",
            ["MCP/stdio call_tool(get_summary)"],
            ["collection_name", "file_name", "blocking", "timeout"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "stdio get_summary did not return expected fields",
        )
        return ok

    @test_case(102, "MCP: Delete Test Collection")
    async def mcp_delete_test_collection(self) -> bool:
        """Delete the MCP test collection and stop SSE/streamable_http servers."""
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.ingestor_server_url}/v1/collections",
                    json=[self.collection],
                ) as resp:
                    delete_ok = resp.status == 200
                    err = None if delete_ok else f"Delete collection failed: {resp.status}"
            stop_sse_ok = self.stop_server()
            # Best-effort: also stop any streamable_http server if it was started
            stream_proc = getattr(self, "stream_proc", None)
            stop_stream_ok = True
            try:
                if stream_proc and stream_proc.poll() is None:
                    stream_proc.terminate()
            except Exception:
                stop_stream_ok = False
            ok = delete_ok and stop_sse_ok and stop_stream_ok
            if not stop_sse_ok or not stop_stream_ok:
                err = (err + "; " if err else "") + "Failed to stop MCP server(s)"
            self.add_test_result(
                102,
                "MCP: Delete Test Collection",
                f"Delete the test collection '{self.collection}' and stop MCP server(s).",
                ["DELETE /v1/collections", "stop_sse_server"],
                ["collection_names"],
                time.time() - start,
                TestStatus.SUCCESS if ok else TestStatus.FAILURE,
                None if ok else "Failed to delete test collection or stop SSE MCP server",
            )
            return ok
        except Exception as e:
            self.add_test_result(
                102,
                "MCP: Delete Test Collection",
                f"Delete the test collection '{self.collection}' and stop MCP server(s).",
                ["DELETE /v1/collections", "stop_sse_server"],
                ["collection_names"],
                time.time() - start,
                TestStatus.FAILURE,
                "Unexpected error during cleanup",
            )
            return False

    @test_case(97, "Start MCP Server (streamable_http)")
    async def start_mcp_server_streamable_http(self) -> bool:
        """Start MCP server in streamable_http mode so subsequent tests can list/call tools."""
        start = time.time()
        try:
            self._start_streamable_http_server()
            # Give the server a brief moment to become ready
            time.sleep(2.0)
            status = TestStatus.SUCCESS
            err = None
        except Exception as e:
            status = TestStatus.FAILURE
            err = str(e)
        self.add_test_result(
            97,
            "Start MCP Server (streamable_http)",
            "Launch MCP server over streamable_http on default FastMCP host/port.",
            ["MCP/streamable_http server"],
            [],
            time.time() - start,
            status,
            None if status == TestStatus.SUCCESS else "streamable_http MCP server did not start successfully",
        )
        return status == TestStatus.SUCCESS

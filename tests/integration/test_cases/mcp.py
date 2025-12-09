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
MCP end-to-end integration sequence
-----------------------------------

This module drives an end-to-end flow against the NVIDIA RAG MCP server using
both transports:
- SSE (http://127.0.0.1:8000/sse)
- streamable_http (http://127.0.0.1:8000/mcp)

Flow overview (numbered tests):
  86) Create a test collection for MCP usage
  87) Upload a sample PDF (with summary generation) to seed the KB
  88) Start MCP server over SSE and wait for readiness
  89) List tools (requires: generate, search, get_summary)
  90) Call 'generate' and assert output contains 'ok'
  91) Call 'search' and assert output mentions 'frost' or 'woods'
  92) Call 'get_summary' and assert output mentions 'frost' or 'woods'
  93) Start MCP server over streamable_http and wait for readiness
  94) List tools (requires: generate, search, get_summary)
  95) Call 'generate' and assert output contains 'ok'
  96) Call 'search' and assert output mentions 'frost' or 'woods'
  97) Call 'get_summary' and assert output mentions 'frost' or 'woods'
  98) Delete the test collection and free the server port
"""

import json
import os
import asyncio
import logging
import shlex
import subprocess
import sys
import time

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case

logger = logging.getLogger(__name__)


class MCPIntegrationModule(BaseTestModule):
    """
    End-to-end MCP integration module.

    Each method corresponds to a numbered test that logs its own result via
    add_test_result. The module prefers small, robust checks (keyword presence)
    to verify returned content without introducing tight coupling to response schemas.
    """

    def __init__(self, test_runner):
        super().__init__(test_runner)
        self.collection = "test_mcp_server"
        self.sse_url = "http://127.0.0.1:8000/sse"
        self.streamable_http_url = "http://127.0.0.1:8000/mcp"

    async def _upload_files(self, files: list[str]) -> bool:
        """
        Upload a small set of files to the Ingestor to prepare the collection.
        Notes:
          - Uses the repository's data/multimodal/woods_frost.pdf by default.
          - Sets generate_summary=True so get_summary can return content.
        """
        if not files:
            logger.warning("No files to upload for MCP tests, continuing without KB.")
            return True
        data = {
            "collection_name": self.collection,
            "blocking": True,
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
        """Launch the MCP server in SSE mode in the background (subprocess)."""
        try:
            self._free_server_port()
        except Exception:
            pass
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        server_path = os.path.join(repo_root, "nvidia_rag_mcp", "mcp_server.py")
        cmd = [sys.executable, server_path, "--transport", "sse"]
        logger.info("Launching SSE MCP server: %s", " ".join(shlex.quote(c) for c in cmd))
        try:
            self.sse_proc = subprocess.Popen(cmd)
            logger.info("SSE server PID: %s", getattr(self.sse_proc, "pid", None))
        except Exception as e:
            logger.error("Failed to start SSE MCP server: %s", e)

    def _start_streamable_http_server(self) -> None:
        """Launch the MCP server in streamable_http mode in the background (subprocess)."""
        try:
            self._free_server_port()
        except Exception:
            pass
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        server_path = os.path.join(repo_root, "nvidia_rag_mcp", "mcp_server.py")
        cmd = [sys.executable, server_path, "--transport", "streamable_http"]
        logger.info("Launching streamable_http MCP server: %s", " ".join(shlex.quote(c) for c in cmd))
        try:
            self.stream_proc = subprocess.Popen(cmd)
            logger.info("streamable_http server PID: %s", getattr(self.stream_proc, "pid", None))
        except Exception as e:
            logger.error("Failed to start streamable_http MCP server: %s", e)

    async def _wait_for_server_ready(self, url, timeout: float = 20.0, interval: float = 0.5) -> bool:
        """
        Poll the given URL until the MCP server is ready or timeout occurs.
        For streamable_http, GET may return HTTP 406 for /mcp; treat 200..299 or 406 as ready.
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as resp:
                        if 200 <= resp.status < 300 or resp.status == 406:
                            return True
            except Exception as e:
                logger.warning("Error waiting for SSE server readiness: %s", e)
            await asyncio.sleep(interval)
        return False

    def _free_server_port(self) -> None:
        """Attempt to kill any process listening on the shared HTTP MCP port."""
        port = 8000
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False, capture_output=True, text=True)
        except Exception as e:
            logger.warning("Error freeing server port: %s", e)
        try:
            out = subprocess.run(["lsof", "-ti", f"tcp:{port}"], check=False, capture_output=True, text=True)
            pids = [p.strip() for p in out.stdout.splitlines() if p.strip().isdigit()]
            for pid in pids:
                try:
                    os.kill(int(pid), 15)
                except Exception as e:
                    logger.warning("Error killing process %s: %s", pid, e)
        except Exception as e:
            logger.warning("Error killing processes: %s", e)

    def _run_mcp_client(self, args: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        client_path = os.path.join(repo_root, "nvidia_rag_mcp", "mcp_client.py")
        mcp_client_cmd = [sys.executable, client_path]
        proc = subprocess.run(mcp_client_cmd + args, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr

    def _parse_listed_tools(self, out: str) -> list[str]:
        """
        Extract tool names from the mcp_client 'list' output.
        Expected lines like 'name: description' or just 'name'.
        """
        tools: list[str] = []
        try:
            for line in (out or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                name = line.split(":", 1)[0].strip()
                if name:
                    tools.append(name)
        except Exception:
            pass
        return tools

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
        """Upload a sample PDF to enable search and summary calls."""
        start = time.time()
        # Reuse a small default file from data dir if available
        default_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
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
        """Start the SSE MCP server and wait until the readiness probe succeeds."""
        start = time.time()
        try:
            self._start_sse_server()
            ready = await self._wait_for_server_ready(self.sse_url, timeout=30.0, interval=1.0)
            status = TestStatus.SUCCESS if ready else TestStatus.FAILURE
        except Exception as e:
            status = TestStatus.FAILURE
            logger.error("Error starting SSE MCP server: %s", e)
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
        """List MCP tools over SSE and require generate/search/get_summary to be present."""
        start = time.time()
        try:
            args = ["list", "--transport", "sse", "--url", self.sse_url]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (SSE list): %s", (out.strip() if out and out.strip() else "<empty>"))
            listed = {t.lower() for t in self._parse_listed_tools(out)}
            required = {"generate", "search", "get_summary"}
            ok = code == 0 and required.issubset(listed)
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error listing tools: %s", e)
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
        """Call 'generate' over SSE and require the output to contain 'ok'."""
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
                self.sse_url,
                "--tool",
                "generate",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (SSE generate): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0 and ("ok" in (out or "").lower())
            _ = None if ok else "SSE generate failed"
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling generate: %s", e)
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
        """Call 'search' over SSE and require the output to mention 'frost' or 'woods'."""
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
                self.sse_url,
                "--tool",
                "search",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (SSE search): %s", (out.strip() if out and out.strip() else "<empty>"))
            out_lc = (out or "").lower()
            ok = code == 0 and ("frost" in out_lc or "woods" in out_lc)
        except Exception as e:
            logger.error("Error calling search: %s", e)
            ok, _ = False, str(e)
            logger.error("Error calling search: %s", e)
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
        """Call 'get_summary' over SSE and require the output to mention 'frost' or 'woods'."""
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
                self.sse_url,
                "--tool",
                "get_summary",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (SSE get_summary): %s", (out.strip() if out and out.strip() else "<empty>"))
            out_lc = (out or "").lower()
            ok = code == 0 and ("frost" in out_lc or "woods" in out_lc)
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling get_summary: %s", e)
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

    @test_case(93, "Start MCP Server (streamable_http)")
    async def start_mcp_server_streamable_http(self) -> bool:
        """Start the streamable_http MCP server and wait until the readiness probe succeeds."""
        start = time.time()
        try:
            self._start_streamable_http_server()
            ready = await self._wait_for_server_ready(self.streamable_http_url, timeout=30.0, interval=1.0)
            status = TestStatus.SUCCESS if ready else TestStatus.FAILURE
        except Exception as e:
            status = TestStatus.FAILURE
            logger.error("Error starting streamable_http MCP server: %s", e)
        self.add_test_result(
            93,
            "Start MCP Server (streamable_http)",
            "Launch MCP server over streamable_http on default FastMCP host/port.",
            ["MCP/streamable_http server"],
            [],
            time.time() - start,
            status,
            None if status == TestStatus.SUCCESS else "streamable_http MCP server did not start successfully",
        )
        return status == TestStatus.SUCCESS

    @test_case(94, "streamable_http: List Tools")
    async def streamable_http_list_tools(self) -> bool:
        """List MCP tools over streamable_http and require generate/search/get_summary to be present."""
        start = time.time()
        try:
            args = [
                "list",
                "--transport",
                "streamable_http",
                "--url",
                self.streamable_http_url,
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (streamable_http list): %s", (out.strip() if out and out.strip() else "<empty>"))
            listed = {t.lower() for t in self._parse_listed_tools(out)}
            required = {"generate", "search", "get_summary"}
            ok = code == 0 and required.issubset(listed)
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error listing tools: %s", e)
        self.add_test_result(
            94,
            "streamable_http: List Tools",
            "List available MCP tools over streamable_http.",
            ["MCP/streamable_http list_tools"],
            [],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http list tools failed",
        )
        return ok

    @test_case(95, "streamable_http: Call Generate")
    async def streamable_http_call_generate(self) -> bool:
        """Call 'generate' over streamable_http and require the output to contain 'ok'."""
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
                self.streamable_http_url,
                "--tool",
                "generate",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (streamable_http generate): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0 and ("ok" in (out or "").lower())
        except Exception as e:
            logger.error("Error calling generate: %s", e)
            ok, _ = False, str(e)
        self.add_test_result(
            95,
            "streamable_http: Call Generate",
            "Call 'generate' tool over streamable_http.",
            ["MCP/streamable_http call_tool(generate)"],
            ["messages", "collection_name"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http generate failed",
        )
        return ok

    @test_case(96, "streamable_http: Call Search")
    async def streamable_http_call_search(self) -> bool:
        """Call 'search' over streamable_http and require the output to mention 'frost' or 'woods'."""
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
                self.streamable_http_url,
                "--tool",
                "search",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (streamable_http search): %s", (out.strip() if out and out.strip() else "<empty>"))
            out_lc = (out or "").lower()
            ok = code == 0 and ("frost" in out_lc or "woods" in out_lc)
        except Exception as e:
            logger.error("Error calling search: %s", e)
            ok = False
        self.add_test_result(
            96,
            "streamable_http: Call Search",
            "Call 'search' tool over streamable_http.",
            ["MCP/streamable_http call_tool(search)"],
            ["query", "collection_name"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http search failed",
        )
        return ok

    @test_case(97, "streamable_http: Call Get Summary")
    async def streamable_http_call_get_summary(self) -> bool:
        """Call 'get_summary' over streamable_http and require the output to mention 'frost' or 'woods'."""
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
                self.streamable_http_url,
                "--tool",
                "get_summary",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (streamable_http get_summary): %s", (out.strip() if out and out.strip() else "<empty>"))
            out_lc = (out or "").lower()
            ok = code == 0 and ("frost" in out_lc or "woods" in out_lc)
        except Exception as e:
            logger.error("Error calling get_summary: %s", e)
            ok, _ = False, str(e)
        self.add_test_result(
            97,
            "streamable_http: Call Get Summary",
            "Call 'get_summary' tool over streamable_http.",
            ["MCP/streamable_http call_tool(get_summary)"],
            ["collection_name", "file_name", "blocking", "timeout"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http get_summary failed",
        )
        return ok

    @test_case(98, "MCP: Delete Test Collection")
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
                    None if delete_ok else f"Delete collection failed: {resp.status}"
            try:
                self._free_server_port()
            except Exception as e:
                logger.warning("Error freeing server port: %s", e)
            ok = delete_ok
            self.add_test_result(
                98,
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
                98,
                "MCP: Delete Test Collection",
                f"Delete the test collection '{self.collection}' and stop MCP server(s).",
                ["DELETE /v1/collections", "stop_sse_server"],
                ["collection_names"],
                time.time() - start,
                TestStatus.FAILURE,
                "Unexpected error during cleanup",
            )
            return False

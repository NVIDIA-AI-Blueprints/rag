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
three transports:
- SSE (http://127.0.0.1:8000/sse)
- streamable_http (http://127.0.0.1:8000/mcp)
- stdio (local process over stdio)

Flow overview (numbered tests):
  86) Start MCP server over SSE and wait for readiness
  87) SSE (Ingestor): Create Collections
  88) SSE (Ingestor): Upload Documents
  89) SSE: List Tools (RAG + Ingestor tools present)
  90) SSE: Call 'generate' (expects 'ok')
  91) SSE: Call 'search' (expects 'frost' or 'woods')
  92) SSE: Call 'get_summary' (expects 'frost' or 'woods')
  93) SSE (Ingestor): Delete Collections
  94) Start MCP server over streamable_http and wait for readiness
  95) streamable_http (Ingestor): Create Collections
  96) streamable_http (Ingestor): Upload Documents
  97) streamable_http: List Tools (RAG + Ingestor tools present)
  98) streamable_http: Call 'generate' (expects 'ok')
  100) streamable_http: Call 'search' (expects 'frost' or 'woods')
  101) streamable_http: Call 'get_summary' (expects 'frost' or 'woods')
  102) streamable_http (Ingestor): Delete Collections
  103) stdio (Ingestor): Create Collections
  104) stdio (Ingestor): Upload Documents
  105) stdio: List Tools (RAG + Ingestor tools present)
  106) stdio: Call 'generate' (expects 'ok')
  107) stdio: Call 'search' (expects 'frost' or 'woods')
  108) stdio: Call 'get_summary' (expects 'frost' or 'woods')
  109) stdio (Ingestor): Delete Collections
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

# Full expected MCP tool surface (Retriever + Ingestor) that should be exposed
# consistently across all transports.
EXPECTED_MCP_TOOLS: set[str] = {
    "generate",
    "search",
    "get_summary",
    "get_documents",
    "delete_documents",
    "update_documents",
    "list_collections",
    "update_collection_metadata",
    "update_document_metadata",
    "create_collections",
    "delete_collections",
    "upload_documents",
}


class MCPIntegrationModule(BaseTestModule):
    """
    End-to-end MCP integration module for NVIDIA RAG.

    This suite exercises:
    - Retriever tools (`generate`, `search`, `get_summary`) over all transports
    - Ingestor tools (collections + documents CRUD/metadata) over all transports
    - Tool discovery (`list` RPC) to ensure the MCP server surface matches expectations

    Each method corresponds to a numbered test that logs its own result via
    `add_test_result`. We intentionally use lightweight content checks
    (e.g., presence of key substrings) instead of asserting full JSON schemas
    to keep the tests stable across minor response-shape changes.
    """

    def __init__(self, test_runner):
        super().__init__(test_runner)
        self.collection = "test_mcp_server"
        self.sse_url = "http://127.0.0.1:8000/sse"
        self.streamable_http_url = "http://127.0.0.1:8000/mcp"

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
                    async with session.get(url, timeout=10) as resp:
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

    def _stop_server(self, proc_attr: str) -> None:
        """Stop a server process using the stored handle for explicit termination."""
        proc = getattr(self, proc_attr, None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _run_mcp_client(self, args: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
        """
        Invoke the `mcp_client.py` CLI as a subprocess.

        Returns:
            (exit_code, stdout, stderr)

        Notes:
            - `args` is the CLI argument list *after* the client path.
            - `timeout` guards against hung transports or server bugs.
        """
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

    @test_case(86, "Start MCP Server (SSE)")
    async def _start_mcp_server_sse(self) -> bool:
        """Start the SSE MCP server and wait until the readiness probe succeeds."""
        logger.info("\n=== Test 86: Start MCP Server (SSE) ===")
        start = time.time()
        try:
            self._start_sse_server()
            ready = await self._wait_for_server_ready(self.sse_url, timeout=30.0, interval=1.0)
            status = TestStatus.SUCCESS if ready else TestStatus.FAILURE
        except Exception as e:
            status = TestStatus.FAILURE
            logger.error("Error starting SSE MCP server: %s", e)
        self.add_test_result(
            86,
            "Start MCP Server (SSE)",
            "Launch MCP server over SSE on http://127.0.0.1:8000.",
            ["MCP/SSE server"],
            [],
            time.time() - start,
            status,
            None if status == TestStatus.SUCCESS else "SSE MCP server did not become ready in time",
        )
        return status == TestStatus.SUCCESS

    @test_case(87, "SSE (Ingestor): Create Collections")
    async def _sse_ingestor_create_collections(self) -> bool:
        """
        Call `create_collections` ingestor tool over SSE.

        This primes the environment by ensuring the test collection exists
        before subsequent upload/search/summary tests run.
        """
        start = time.time()
        try:
            payload = {"collection_names": [self.collection]}
            args = [
                "call",
                "--transport",
                "sse",
                "--url",
                self.sse_url,
                "--tool",
                "create_collections",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (SSE create_collections): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling create_collections (SSE): %s", e)
        self.add_test_result(
            87,
            "SSE (Ingestor): Create Collections",
            "Create collection(s) using ingestor tool over SSE.",
            ["MCP/SSE call_tool(create_collections)"],
            ["collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "SSE ingestor create_collections failed",
        )
        return ok

    @test_case(88, "SSE (Ingestor): Upload Documents")
    async def _sse_ingestor_upload_documents(self) -> bool:
        """
        Call `upload_documents` ingestor tool over SSE.

        Uses the sample `woods_frost.pdf` multimodal document so that later
        RAG `search` and `get_summary` tests can assert against known content.
        """
        start = time.time()
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            pdf_path = os.path.join(repo_root, "data", "multimodal", "woods_frost.pdf")
            payload = {
                "collection_name": self.collection,
                "file_paths": [pdf_path],
                "blocking": True,
                "generate_summary": True,
                "split_options": {"chunk_size": 512, "chunk_overlap": 150},
            }
            args = [
                "call",
                "--transport",
                "sse",
                "--url",
                self.sse_url,
                "--tool",
                "upload_documents",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (SSE upload_documents): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling upload_documents (SSE): %s", e)
        self.add_test_result(
            88,
            "SSE (Ingestor): Upload Documents",
            "Upload document(s) using ingestor tool over SSE.",
            ["MCP/SSE call_tool(upload_documents)"],
            ["collection_name", "file_paths", "blocking", "generate_summary"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "SSE ingestor upload_documents failed",
        )
        return ok

    @test_case(89, "SSE: List Tools")
    async def _sse_list_tools(self) -> bool:
        """
        List MCP tools over SSE and verify all expected tools are exposed.

        The expectation is that both Retriever tools and all Ingestor tools are
        registered with FastMCP, matching the server implementation in
        `nvidia_rag_mcp/mcp_server.py`.
        """
        logger.info("\n=== Test 89: SSE: List Tools ===")
        start = time.time()
        missing = []
        try:
            args = ["list", "--transport", "sse", "--url", self.sse_url]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (SSE list): %s", (out.strip() if out and out.strip() else "<empty>"))
            listed = {t.lower() for t in self._parse_listed_tools(out)}
            missing = sorted(EXPECTED_MCP_TOOLS - listed)
            ok = code == 0 and not missing
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
            None if ok else f"SSE list tools did not include all required tools; missing={missing}",
        )
        return ok

    @test_case(90, "SSE: Call Generate")
    async def _sse_call_generate(self) -> bool:
        """
        Call `generate` over SSE and require the output to contain 'ok'.

        This validates that the RAG pipeline is wired correctly end‑to‑end via MCP,
        including streaming handling and basic prompt forwarding.
        """
        logger.info("\n=== Test 90: SSE: Call Generate ===")
        start = time.time()
        try:
            payload = {
                "messages": [{"role": "user", "content": "Say 'ok'"}],
                "collection_names": [self.collection],
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
            ["messages", "collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "SSE generate did not return expected content",
        )
        return ok

    @test_case(91, "SSE: Call Search")
    async def _sse_call_search(self) -> bool:
        """
        Call `search` over SSE and require the output to mention 'frost' or 'woods'.

        The test asserts that the search pipeline can retrieve citations from the
        previously ingested `woods_frost.pdf` collection.
        """
        logger.info("\n=== Test 91: SSE: Call Search ===")
        start = time.time()
        try:
            payload = {
                "query": "woods frost",
                "collection_names": [self.collection],
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
        self.add_test_result(
            91,
            "SSE: Call Search",
            "Call 'search' tool over SSE.",
            ["MCP/SSE call_tool(search)"],
            ["query", "collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "SSE search did not return results",
        )
        return ok

    @test_case(92, "SSE: Call Get Summary")
    async def _sse_call_get_summary(self) -> bool:
        """
        Call `get_summary` over SSE and require the output to mention 'frost' or 'woods'.

        This verifies that document summaries can be retrieved via MCP and that
        the ingestor‑generated summaries are accessible to clients.
        """
        logger.info("\n=== Test 92: SSE: Call Get Summary ===")
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

    @test_case(93, "SSE (Ingestor): Delete Collections")
    async def _sse_ingestor_delete_collections(self) -> bool:
        """Call 'delete_collections' ingestor tool over SSE."""
        start = time.time()
        try:
            payload = {"collection_names": [self.collection]}
            args = [
                "call",
                "--transport",
                "sse",
                "--url",
                self.sse_url,
                "--tool",
                "delete_collections",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (SSE delete_collections): %s", (out.strip() if out and out.strip() else "<empty>"))
            self._stop_server("sse_proc")
            self._free_server_port()
            ok = code == 0
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling delete_collections (SSE): %s", e)
        self.add_test_result(
            93,
            "SSE (Ingestor): Delete Collections",
            "Delete collection(s) using ingestor tool over SSE.",
            ["MCP/SSE call_tool(delete_collections)"],
            ["collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "SSE ingestor delete_collections failed",
        )
        return ok

    @test_case(94, "Start MCP Server (streamable_http)")
    async def _start_mcp_server_streamable_http(self) -> bool:
        """Start the streamable_http MCP server and wait until the readiness probe succeeds."""
        logger.info("\n=== Test 94: Start MCP Server (streamable_http) ===")
        start = time.time()
        try:
            self._start_streamable_http_server()
            ready = await self._wait_for_server_ready(self.streamable_http_url, timeout=30.0, interval=1.0)
            status = TestStatus.SUCCESS if ready else TestStatus.FAILURE
        except Exception as e:
            status = TestStatus.FAILURE
            logger.error("Error starting streamable_http MCP server: %s", e)
        self.add_test_result(
            94,
            "Start MCP Server (streamable_http)",
            "Launch MCP server over streamable_http on default FastMCP host/port.",
            ["MCP/streamable_http server"],
            [],
            time.time() - start,
            status,
            None if status == TestStatus.SUCCESS else "streamable_http MCP server did not start successfully",
        )
        return status == TestStatus.SUCCESS

    @test_case(95, "streamable_http (Ingestor): Create Collections")
    async def _streamable_http_ingestor_create_collections(self) -> bool:
        """Call 'create_collections' ingestor tool over streamable_http."""
        start = time.time()
        try:
            payload = {"collection_names": [self.collection]}
            args = [
                "call",
                "--transport",
                "streamable_http",
                "--url",
                self.streamable_http_url,
                "--tool",
                "create_collections",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (streamable_http create_collections): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling create_collections (streamable_http): %s", e)
        self.add_test_result(
            95,
            "streamable_http (Ingestor): Create Collections",
            "Create collection(s) using ingestor tool over streamable_http.",
            ["MCP/streamable_http call_tool(create_collections)"],
            ["collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http ingestor create_collections failed",
        )
        return ok

    @test_case(96, "streamable_http (Ingestor): Upload Documents")
    async def _streamable_http_ingestor_upload_documents(self) -> bool:
        """Call 'upload_documents' ingestor tool over streamable_http."""
        start = time.time()
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            pdf_path = os.path.join(repo_root, "data", "multimodal", "woods_frost.pdf")
            payload = {
                "collection_name": self.collection,
                "file_paths": [pdf_path],
                "blocking": True,
                "generate_summary": True,
                "split_options": {"chunk_size": 512, "chunk_overlap": 150},
            }
            args = [
                "call",
                "--transport",
                "streamable_http",
                "--url",
                self.streamable_http_url,
                "--tool",
                "upload_documents",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (streamable_http upload_documents): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling upload_documents (streamable_http): %s", e)
        self.add_test_result(
            96,
            "streamable_http (Ingestor): Upload Documents",
            "Upload document(s) using ingestor tool over streamable_http.",
            ["MCP/streamable_http call_tool(upload_documents)"],
            ["collection_name", "file_paths", "blocking", "generate_summary"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http ingestor upload_documents failed",
        )
        return ok

    @test_case(97, "streamable_http: List Tools")
    async def _streamable_http_list_tools(self) -> bool:
        """
        List MCP tools over streamable_http and verify all expected tools are exposed.

        This mirrors `_sse_list_tools` but exercises the FastMCP streamable_http
        transport instead of raw SSE, ensuring the tool surface is identical.
        """
        logger.info("\n=== Test 97: streamable_http: List Tools ===")
        start = time.time()
        missing = []
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
            missing = sorted(EXPECTED_MCP_TOOLS - listed)
            ok = code == 0 and not missing
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error listing tools: %s", e)
        self.add_test_result(
            97,
            "streamable_http: List Tools",
            "List available MCP tools over streamable_http.",
            ["MCP/streamable_http list_tools"],
            [],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else f"streamable_http list tools failed; missing={missing}",
        )
        return ok

    @test_case(98, "streamable_http: Call Generate")
    async def _streamable_http_call_generate(self) -> bool:
        """Call 'generate' over streamable_http and require the output to contain 'ok'."""
        logger.info("\n=== Test 98: streamable_http: Call Generate ===")
        start = time.time()
        try:
            payload = {
                "messages": [{"role": "user", "content": "Say 'ok'"}],
                "collection_names": [self.collection],
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
            98,
            "streamable_http: Call Generate",
            "Call 'generate' tool over streamable_http.",
            ["MCP/streamable_http call_tool(generate)"],
            ["messages", "collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http generate failed",
        )
        return ok

    @test_case(100, "streamable_http: Call Search")
    async def _streamable_http_call_search(self) -> bool:
        """Call 'search' over streamable_http and require the output to mention 'frost' or 'woods'."""
        logger.info("\n=== Test 100: streamable_http: Call Search ===")
        start = time.time()
        try:
            payload = {
                "query": "woods frost",
                "collection_names": [self.collection],
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
            100,
            "streamable_http: Call Search",
            "Call 'search' tool over streamable_http.",
            ["MCP/streamable_http call_tool(search)"],
            ["query", "collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http search failed",
        )
        return ok

    @test_case(101, "streamable_http: Call Get Summary")
    async def _streamable_http_call_get_summary(self) -> bool:
        """Call 'get_summary' over streamable_http and require the output to mention 'frost' or 'woods'."""
        logger.info("\n=== Test 101: streamable_http: Call Get Summary ===")
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

    @test_case(102, "streamable_http (Ingestor): Delete Collections")
    async def _streamable_http_ingestor_delete_collections(self) -> bool:
        """Call 'delete_collections' ingestor tool over streamable_http."""
        start = time.time()
        try:
            payload = {"collection_names": [self.collection]}
            args = [
                "call",
                "--transport",
                "streamable_http",
                "--url",
                self.streamable_http_url,
                "--tool",
                "delete_collections",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (streamable_http delete_collections): %s", (out.strip() if out and out.strip() else "<empty>"))
            self._stop_server("stream_proc")
            self._free_server_port()
            ok = code == 0
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling delete_collections (streamable_http): %s", e)
        self.add_test_result(
            102,
            "streamable_http (Ingestor): Delete Collections",
            "Delete collection(s) using ingestor tool over streamable_http.",
            ["MCP/streamable_http call_tool(delete_collections)"],
            ["collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "streamable_http ingestor delete_collections failed",
        )
        return ok

    @test_case(103, "stdio (Ingestor): Create Collections")
    async def _stdio_ingestor_create_collections(self) -> bool:
        """Call 'create_collections' ingestor tool over stdio."""
        start = time.time()
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            server_path = os.path.join(repo_root, "nvidia_rag_mcp", "mcp_server.py")
            payload = {"collection_names": [self.collection]}
            args = [
                "call",
                "--transport",
                "stdio",
                "--command",
                sys.executable,
                "--args",
                f"{server_path} --transport stdio",
                "--tool",
                "create_collections",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (stdio create_collections): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling create_collections (stdio): %s", e)
        self.add_test_result(
            103,
            "stdio (Ingestor): Create Collections",
            "Create collection(s) using ingestor tool over stdio.",
            ["MCP/stdio call_tool(create_collections)"],
            ["collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "stdio ingestor create_collections failed",
        )
        return ok

    @test_case(104, "stdio (Ingestor): Upload Documents")
    async def _stdio_ingestor_upload_documents(self) -> bool:
        """Call 'upload_documents' ingestor tool over stdio."""
        start = time.time()
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            server_path = os.path.join(repo_root, "nvidia_rag_mcp", "mcp_server.py")
            pdf_path = os.path.join(repo_root, "data", "multimodal", "woods_frost.pdf")
            payload = {
                "collection_name": self.collection,
                "file_paths": [pdf_path],
                "blocking": True,
                "generate_summary": True,
                "split_options": {"chunk_size": 512, "chunk_overlap": 150},
            }
            args = [
                "call",
                "--transport",
                "stdio",
                "--command",
                sys.executable,
                "--args",
                f"{server_path} --transport stdio",
                "--tool",
                "upload_documents",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (stdio upload_documents): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling upload_documents (stdio): %s", e)
        self.add_test_result(
            104,
            "stdio (Ingestor): Upload Documents",
            "Upload document(s) using ingestor tool over stdio.",
            ["MCP/stdio call_tool(upload_documents)"],
            ["collection_name", "file_paths", "blocking", "generate_summary"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "stdio ingestor upload_documents failed",
        )
        return ok

    @test_case(105, "stdio: List Tools")
    async def _stdio_list_tools(self) -> bool:
        """
        List MCP tools over stdio and verify all expected tools are exposed.

        This validates the stdio transport path (used for local MCP servers),
        asserting that the same RAG + Ingestor tools are available as over HTTP.
        """
        start = time.time()
        missing = []
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            server_path = os.path.join(repo_root, "nvidia_rag_mcp", "mcp_server.py")
            args = [
                "list",
                "--transport",
                "stdio",
                "--command",
                sys.executable,
                "--args",
                f"{server_path} --transport stdio",
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (stdio list): %s", (out.strip() if out and out.strip() else "<empty>"))
            listed = {t.lower() for t in self._parse_listed_tools(out)}
            missing = sorted(EXPECTED_MCP_TOOLS - listed)
            ok = code == 0 and not missing
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error listing tools (stdio): %s", e)
        self.add_test_result(
            105,
            "stdio: List Tools",
            "List available MCP tools over stdio.",
            ["MCP/stdio list_tools"],
            [],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else f"stdio list tools did not include all required tools; missing={missing}",
        )
        return ok

    @test_case(106, "stdio: Call Generate")
    async def _stdio_call_generate(self) -> bool:
        """Call 'generate' over stdio and require the output to contain 'ok'."""
        start = time.time()
        try:
            payload = {
                "messages": [{"role": "user", "content": "Say 'ok'"}],
                "collection_names": [self.collection],
            }
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            server_path = os.path.join(repo_root, "nvidia_rag_mcp", "mcp_server.py")
            args = [
                "call",
                "--transport",
                "stdio",
                "--command",
                sys.executable,
                "--args",
                f"{server_path} --transport stdio",
                "--tool",
                "generate",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (stdio generate): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0 and ("ok" in (out or "").lower())
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling generate (stdio): %s", e)
        self.add_test_result(
            106,
            "stdio: Call Generate",
            "Call 'generate' tool over stdio.",
            ["MCP/stdio call_tool(generate)"],
            ["messages", "collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "stdio generate did not return expected content",
        )
        return ok

    @test_case(107, "stdio: Call Search")
    async def _stdio_call_search(self) -> bool:
        """Call 'search' over stdio and require the output to mention 'frost' or 'woods'."""
        start = time.time()
        try:
            payload = {
                "query": "woods frost",
                "collection_names": [self.collection],
            }
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            server_path = os.path.join(repo_root, "nvidia_rag_mcp", "mcp_server.py")
            args = [
                "call",
                "--transport",
                "stdio",
                "--command",
                sys.executable,
                "--args",
                f"{server_path} --transport stdio",
                "--tool",
                "search",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (stdio search): %s", (out.strip() if out and out.strip() else "<empty>"))
            out_lc = (out or "").lower()
            ok = code == 0 and ("frost" in out_lc or "woods" in out_lc)
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling search (stdio): %s", e)
        self.add_test_result(
            107,
            "stdio: Call Search",
            "Call 'search' tool over stdio.",
            ["MCP/stdio call_tool(search)"],
            ["query", "collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "stdio search did not return results",
        )
        return ok

    @test_case(108, "stdio: Call Get Summary")
    async def _stdio_call_get_summary(self) -> bool:
        """Call 'get_summary' over stdio and require the output to mention 'frost' or 'woods'."""
        start = time.time()
        try:
            payload = {
                "collection_name": self.collection,
                "file_name": "woods_frost.pdf",
                "blocking": False,
                "timeout": 60,
            }
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            server_path = os.path.join(repo_root, "nvidia_rag_mcp", "mcp_server.py")
            args = [
                "call",
                "--transport",
                "stdio",
                "--command",
                sys.executable,
                "--args",
                f"{server_path} --transport stdio",
                "--tool",
                "get_summary",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (stdio get_summary): %s", (out.strip() if out and out.strip() else "<empty>"))
            out_lc = (out or "").lower()
            ok = code == 0 and ("frost" in out_lc or "woods" in out_lc)
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling get_summary (stdio): %s", e)
        self.add_test_result(
            108,
            "stdio: Call Get Summary",
            "Call 'get_summary' tool over stdio.",
            ["MCP/stdio call_tool(get_summary)"],
            ["collection_name", "file_name", "blocking", "timeout"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "stdio get_summary did not return expected fields",
        )
        return ok

    @test_case(109, "stdio (Ingestor): Delete Collections")
    async def _stdio_ingestor_delete_collections(self) -> bool:
        """Call 'delete_collections' ingestor tool over stdio."""
        start = time.time()
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            server_path = os.path.join(repo_root, "nvidia_rag_mcp", "mcp_server.py")
            payload = {"collection_names": [self.collection]}
            args = [
                "call",
                "--transport",
                "stdio",
                "--command",
                sys.executable,
                "--args",
                f"{server_path} --transport stdio",
                "--tool",
                "delete_collections",
                "--json-args",
                json.dumps(payload),
            ]
            code, out, _ = self._run_mcp_client(args)
            logger.info("MCP client output (stdio delete_collections): %s", (out.strip() if out and out.strip() else "<empty>"))
            ok = code == 0
        except Exception as e:
            ok, _ = False, str(e)
            logger.error("Error calling delete_collections (stdio): %s", e)
        self.add_test_result(
            109,
            "stdio (Ingestor): Delete Collections",
            "Delete collection(s) using ingestor tool over stdio.",
            ["MCP/stdio call_tool(delete_collections)"],
            ["collection_names"],
            time.time() - start,
            TestStatus.SUCCESS if ok else TestStatus.FAILURE,
            None if ok else "stdio ingestor delete_collections failed",
        )
        return ok

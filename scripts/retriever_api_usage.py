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
NVIDIA RAG Retriever API CLI

Examples
- Generate with a simple query:
  python scripts/retriever_api_usage.py "Compare bluetooth speaker with hammer"

- Search only with a query:
  python scripts/retriever_api_usage.py --mode search "Tell me about robert frost's poems"

- Generate with a specific collection:
  python scripts/retriever_api_usage.py \
    --payload-json '{"collection_names":["my_collection"]}' \
    "What is RAG?"

- Search with a specific collection:
  python scripts/retriever_api_usage.py \
    --mode search \
    --payload-json '{"collection_names":["my_collection"]}' \
    "Tell me about robert frost's poems"

- Generate using a raw JSON payload string:
  python scripts/retriever_api_usage.py \
    --payload-json '{"messages":[{"role":"user","content":"Your query here"}],"use_knowledge_base":true,"temperature":0.2,"top_p":0.7,"max_tokens":1024,"reranker_top_k":2,"vdb_top_k":10,"vdb_endpoint":"http://milvus:19530","collection_names":["multimodal_data"],"enable_query_rewriting":true,"enable_reranker":true,"enable_citations":true,"stop":[],"filter_expr":""}'

- Search using a payload file and save output to JSON:
  python scripts/retriever_api_usage.py \
    --mode search \
    --payload-file payloads/search.json \
    --output-json out/search_result.json

- Override host (defaults to http://localhost:8081):
  python scripts/retriever_api_usage.py \
    --host http://my-host:8081 \
    "What's the difference between product A and product B?"

Notes
- You can pass a partial payload with --payload-json/--payload-file. Only the provided
  fields will override defaults; all other fields use sensible defaults.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

import aiohttp

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_HOST = "localhost"
DEFAULT_PORT = "8081"


class LogfmtFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__()

    def formatTime(self, record, datefmt=None):
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record)
        level = record.levelname
        logger_name = record.name
        # Ensure the message is safely quoted
        try:
            message = record.getMessage()
        except Exception:
            message = record.msg if isinstance(record.msg, str) else str(record.msg)
        # Use JSON to ensure proper escaping and quotes
        message_json = json.dumps(message, ensure_ascii=False)
        return f"{ts} | {level} | {logger_name} | {message_json}"


def configure_logging() -> None:
    root = logging.getLogger()
    # Reset existing handlers for consistent formatting
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(LogfmtFormatter())
    root.addHandler(handler)


logger = logging.getLogger("retriever_cli")


# -----------------------------------------------------------------------------
# Payload builders (kept constant for now; structured for easy future args)
# -----------------------------------------------------------------------------


def deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two dicts without mutating inputs.

    - Dicts are merged recursively
    - Lists and scalars in overrides replace base
    """

    def _merge(a: Any, b: Any) -> Any:
        if isinstance(a, dict) and isinstance(b, dict):
            result: dict[str, Any] = {}
            for key in set(a.keys()) | set(b.keys()):
                if key in a and key in b:
                    result[key] = _merge(a[key], b[key])
                elif key in b:
                    result[key] = b[key]
                else:
                    result[key] = a[key]
            return result
        # default: replace
        return b if b is not None else a

    return _merge(dict(base), dict(overrides))


def build_generate_payload(query: str) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ],
        "use_knowledge_base": True,
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
        "reranker_top_k": 2,
        "vdb_top_k": 10,
        "vdb_endpoint": "http://milvus:19530",
        "collection_names": ["multimodal_data"],
        "enable_query_rewriting": True,
        "enable_reranker": True,
        "enable_citations": True,
        "stop": [],
        "filter_expr": "",
        # Model/endpoint overrides intentionally omitted (constants for now)
    }


def build_search_payload(query: str) -> dict[str, Any]:
    return {
        "query": query,
        "reranker_top_k": 2,
        "vdb_top_k": 10,
        "vdb_endpoint": "http://milvus:19530",
        "collection_names": ["multimodal_data"],
        "messages": [],
        "enable_query_rewriting": False,
        "enable_reranker": True,
        # Optional: "filter_expr": "",
    }


# -----------------------------------------------------------------------------
# Client
# -----------------------------------------------------------------------------


class RagClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._http_timeout = aiohttp.ClientTimeout(total=300)

    async def _print_json_response(self, response: aiohttp.ClientResponse) -> None:
        try:
            response_json = await response.json()
            print(json.dumps(response_json, indent=2))
        except aiohttp.ContentTypeError:
            text = await response.text()
            logger.error("Unexpected non-JSON response: %s", text)

    async def search(
        self,
        query: str | None = None,
        payload: dict[str, Any] | None = None,
        output_path: str | None = None,
    ) -> int:
        url = f"{self.base_url}/v1/search"
        # Build defaults and apply partial overrides if provided
        if payload is None:
            if not query:
                logger.error("Search requires a query or a payload containing 'query'")
                return 2
            final_payload = build_search_payload(query)
        else:
            seed_query = query if query else payload.get("query")
            if not seed_query:
                logger.error("Search requires a query or a payload containing 'query'")
                return 2
            defaults = build_search_payload(seed_query)
            final_payload = deep_merge_dicts(defaults, payload)
        logger.info("POST %s", url)

        async with aiohttp.ClientSession(timeout=self._http_timeout) as session:
            try:
                async with session.post(url=url, json=final_payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(
                            "Search request failed: HTTP %s | %s", response.status, text
                        )
                        return 1
                    try:
                        response_json = await response.json()
                    except aiohttp.ContentTypeError:
                        text = await response.text()
                        logger.error("Unexpected non-JSON response: %s", text)
                        return 1

                    if output_path:
                        try:
                            parent = os.path.dirname(output_path)
                            if parent:
                                os.makedirs(parent, exist_ok=True)
                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(
                                    response_json, f, indent=2, ensure_ascii=False
                                )
                            logger.info("Saved response to %s", output_path)
                        except Exception as exc:
                            logger.error(
                                "Failed to write output JSON '%s': %s", output_path, exc
                            )
                            return 1
                    else:
                        print(json.dumps(response_json, indent=2))
                    return 0
            except aiohttp.ClientError as exc:
                logger.exception("Network error during search: %s", exc)
                return 1

    async def generate(
        self,
        query: str | None = None,
        payload: dict[str, Any] | None = None,
        output_path: str | None = None,
    ) -> int:
        url = f"{self.base_url}/v1/generate"
        # Build defaults and apply partial overrides if provided
        if payload is None:
            if not query:
                logger.error(
                    "Generate requires a query or a payload containing 'messages'"
                )
                return 2
            final_payload = build_generate_payload(query)
        else:
            defaults = build_generate_payload(query or "")
            final_payload = deep_merge_dicts(defaults, payload)
            # Ensure messages are present in the final payload
            if not final_payload.get("messages"):
                logger.error(
                    "Generate requires 'messages' in payload when no query is provided"
                )
                return 2
        logger.info("POST %s", url)

        async with aiohttp.ClientSession(timeout=self._http_timeout) as session:
            try:
                async with session.post(url=url, json=final_payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(
                            "Generate request failed: HTTP %s | %s",
                            response.status,
                            text,
                        )
                        return 1
                    await self._stream_generate_response(
                        response, output_path=output_path
                    )
                    return 0
            except aiohttp.ClientError as exc:
                logger.exception("Network error during generate: %s", exc)
                return 1

    async def _stream_generate_response(
        self, response: aiohttp.ClientResponse, output_path: str | None = None
    ) -> None:
        buffer = ""
        citations_logged = False
        collected_citations: Any | None = None
        collected_text_parts: list[str] = []
        final_metrics: dict[str, Any] = {}

        async for chunk in response.content.iter_chunked(8192):
            if not chunk:
                continue
            try:
                decoded = chunk.decode("utf-8")
            except UnicodeDecodeError:
                # Skip undecodable chunk
                continue
            buffer += decoded

            lines = buffer.split("\n")
            buffer = lines[-1]

            for line in lines[:-1]:
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                json_str = line[6:].strip()
                if not json_str:
                    continue
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    # Malformed JSON piece; skip
                    continue

                # Optional one-time citations log (if provided on first chunk)
                if not citations_logged and "citations" in data:
                    citations_logged = True
                    collected_citations = data.get("citations", [])
                    logger.info(
                        "Citations received (%d)", len(data.get("citations", []))
                    )

                # Stream message content
                message = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                if message:
                    if output_path:
                        collected_text_parts.append(message)
                    else:
                        # Stream to stdout without logger formatting
                        print(message, end="", flush=True)

                finish_reason: str | None = data.get("choices", [{}])[0].get(
                    "finish_reason"
                )
                if finish_reason == "stop":
                    if not output_path:
                        print()  # newline after final content
                    metrics: dict[str, Any] = data.get("metrics", {})
                    # Professional, concise metrics summary
                    logger.info(
                        "RAG metrics | llm_generation_ms=%s ttft_ms=%s reranker_ms=%s retrieval_ms=%s rag_ttft_ms=%s",
                        metrics.get("llm_generation_time_ms", "N/A"),
                        metrics.get("llm_ttft_ms", "N/A"),
                        metrics.get("context_reranker_time_ms", "N/A"),
                        metrics.get("retrieval_time_ms", "N/A"),
                        metrics.get("rag_ttft_ms", "N/A"),
                    )
                    final_metrics = metrics
                    # If output path provided, persist structured result
                    if output_path:
                        try:
                            output_obj: dict[str, Any] = {
                                "content": "".join(collected_text_parts),
                                "metrics": final_metrics,
                            }
                            if collected_citations is not None:
                                output_obj["citations"] = collected_citations
                            parent = os.path.dirname(output_path)
                            if parent:
                                os.makedirs(parent, exist_ok=True)
                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(output_obj, f, indent=2, ensure_ascii=False)
                            logger.info("Saved response to %s", output_path)
                        except Exception as exc:
                            logger.error(
                                "Failed to write output JSON '%s': %s", output_path, exc
                            )
                    return


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query NVIDIA RAG services to either generate an answer or retrieve search results."
        )
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Optional natural-language query. Ignored if --payload-* is provided.",
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "search"],
        default="generate",
        help=(
            "Operation mode: 'generate' streams an answer; 'search' returns top documents."
        ),
    )
    parser.add_argument(
        "--host",
        default=f"http://{DEFAULT_HOST}:{DEFAULT_PORT}",
        help="Base URL of the RAG server, e.g. http://localhost:8081",
    )
    payload_group = parser.add_mutually_exclusive_group()
    payload_group.add_argument(
        "--payload-json",
        dest="payload_json",
        help="Raw JSON payload string to send to the endpoint.",
    )
    payload_group.add_argument(
        "--payload-file",
        dest="payload_file",
        help="Path to JSON file containing the payload to send.",
    )
    parser.add_argument(
        "--output-json",
        dest="output_json",
        help="Path to save the response JSON. If omitted, prints to stdout.",
    )
    return parser.parse_args(argv)


async def async_main() -> int:
    configure_logging()
    args = parse_args()

    base_url = args.host.rstrip("/")
    logger.info("Starting request | mode=%s base_url=%s", args.mode, base_url)

    client = RagClient(base_url)

    # Resolve payload precedence (file > json > builder)
    payload: dict[str, Any] | None = None
    if args.payload_file:
        try:
            with open(args.payload_file, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.error("Failed to read payload file '%s': %s", args.payload_file, exc)
            return 2
    elif args.payload_json:
        try:
            payload = json.loads(args.payload_json)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in --payload-json: %s", exc)
            return 2

    if args.mode == "search":
        return await client.search(
            query=args.query, payload=payload, output_path=args.output_json
        )
    return await client.generate(
        query=args.query, payload=payload, output_path=args.output_json
    )


def main() -> None:
    try:
        exit_code = asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        exit_code = 130
    except Exception:
        logger.exception("Unhandled error")
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

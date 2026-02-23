# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Nemotron Parse API: image (base64) → structured elements.

Vendored for ingestor-server; includes render_and_parse_page_impl for sequential path.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import requests

try:
    import ray
except ImportError:
    ray = None

RAY_AVAILABLE = ray is not None

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

try:
    from openai import APIConnectionError, APITimeoutError, OpenAI
except ImportError:
    OpenAI = None
    APIConnectionError = Exception
    APITimeoutError = Exception

from nvidia_rag.ingestor_server.ingestion_pipelines.nemotron_parse.extraction.pdf_render import (
    encode_image_to_base64,
    pdfium_render_page_from_doc,
    pdfium_render_page_to_pil,
)

logger = logging.getLogger(__name__)


def is_retryable_request_error(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.ConnectionError):
        return True
    if (
        isinstance(exc, requests.exceptions.HTTPError)
        and getattr(exc, "response", None) is not None
    ):
        return getattr(exc.response, "status_code", 0) in (
            429,
            500,
            502,
            503,
            504,
        )
    if isinstance(exc, requests.exceptions.RequestException):
        return True
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    return False


def parse_page_image_to_elements(
    page_b64: str,
    page_num: int,
    parse_url: str,
    parse_model: str,
    parse_max_tokens: int,
    parse_temperature: float,
    parse_timeout: int,
    api_key: str,
    retry_cfg_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    retry_cfg = SimpleNamespace(
        stop_after_attempt=retry_cfg_dict.get("stop_after_attempt", 3),
        wait_min=retry_cfg_dict.get("wait_min", 2),
        wait_max=retry_cfg_dict.get("wait_max", 30),
    )
    # Base URL only: OpenAI client appends /chat/completions; so strip if env has full path
    url = (parse_url or "").rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")].rstrip("/")
    if not url:
        return []

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{page_b64}"},
                },
            ],
        }
    ]
    tools = [{"type": "function", "function": {"name": "markdown_bbox"}}]

    arguments_str: str | None = None
    if OpenAI is not None:
        try:
            client = OpenAI(
                base_url=url,
                api_key=(api_key or "dummy").strip() or "dummy",
                timeout=float(parse_timeout),
            )

            @retry(
                stop=stop_after_attempt(retry_cfg.stop_after_attempt),
                wait=wait_exponential(
                    multiplier=1, min=retry_cfg.wait_min, max=retry_cfg.wait_max
                ),
                retry=retry_if_exception(is_retryable_request_error),
                reraise=True,
            )
            def _create():
                return client.chat.completions.create(
                    model=parse_model,
                    messages=messages,
                    tools=tools,
                    max_tokens=parse_max_tokens,
                    temperature=parse_temperature,
                )

            completion = _create()
            if completion.choices:
                msg = completion.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    tc = tool_calls[0]
                    arguments_str = (
                        getattr(getattr(tc, "function", None), "arguments", None)
                        or "[]"
                    )
        except Exception as e:
            logger.warning(
                "Nemotron Parse OpenAI path failed (page %s): %s",
                page_num,
                e,
                exc_info=True,
            )

    if arguments_str is None:
        try:
            endpoint = f"{url}/chat/completions"
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = {
                "model": parse_model,
                "messages": messages,
                "max_tokens": parse_max_tokens,
                "temperature": parse_temperature,
                "tools": tools,
            }
            resp = http_post_with_retry(
                endpoint, headers, payload, parse_timeout, retry_cfg
            )
            resp.raise_for_status()
            response_json = resp.json()
            tool_call = (
                response_json.get("choices", [{}])[0]
                .get("message", {})
                .get("tool_calls", [{}])[0]
            )
            if tool_call:
                arguments_str = tool_call.get("function", {}).get("arguments", "[]")
        except Exception:
            return []

    if arguments_str is None:
        return []

    elements: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(arguments_str)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
            elements = parsed[0]
        elif isinstance(parsed, list):
            elements = parsed
    except Exception:
        elements = []

    for e in elements:
        if isinstance(e, dict):
            e["page_id"] = page_num

    return elements


def http_post_with_retry(
    url: str,
    headers: Dict[str, str],
    payload: Any,
    timeout: int,
    retry_cfg: SimpleNamespace,
) -> requests.Response:
    @retry(
        stop=stop_after_attempt(retry_cfg.stop_after_attempt),
        wait=wait_exponential(
            multiplier=1, min=retry_cfg.wait_min, max=retry_cfg.wait_max
        ),
        retry=retry_if_exception(is_retryable_request_error),
        reraise=True,
    )
    def _post() -> requests.Response:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp

    return _post()


def render_and_parse_page_impl(
    pdf_path: str,
    page_index: int,
    page_num: int,
    dpi: int,
    target_size: Tuple[int, int],
    parse_url: str,
    parse_model: str,
    parse_max_tokens: int,
    parse_temperature: float,
    parse_timeout: int,
    api_key: str,
    retry_cfg_dict: Dict[str, Any],
    doc: Any,
) -> Tuple[int, str, List[Dict[str, Any]]]:
    if not (parse_url and (parse_url or "").strip()):
        return (page_num, "", [])

    if doc is not None:
        try:
            img = pdfium_render_page_from_doc(
                doc, page_index, dpi, target_size, pdf_path
            )
        except Exception:
            return (page_num, "", [])
    else:
        try:
            img = pdfium_render_page_to_pil(pdf_path, page_index, dpi, target_size)
        except Exception:
            return (page_num, "", [])

    page_b64 = encode_image_to_base64(img, fmt="PNG")
    elements = parse_page_image_to_elements(
        page_b64,
        page_num,
        parse_url,
        parse_model,
        parse_max_tokens,
        parse_temperature,
        parse_timeout,
        api_key,
        retry_cfg_dict,
    )
    return (page_num, page_b64, elements)


def render_and_parse_page_task(
    pdf_path: str,
    page_index: int,
    page_num: int,
    dpi: int,
    target_size: Tuple[int, int],
    parse_urls: List[str],
    parse_model: str,
    parse_max_tokens: int,
    parse_temperature: float,
    parse_timeout: int,
    api_key: str,
    retry_cfg_dict: Dict[str, Any],
) -> Tuple[int, str, List[Dict[str, Any]]]:
    """
    Ray remote task: open PDF for this page, render, parse. No shared doc.
    parse_urls: list of base URLs; URL chosen by page_index % len(parse_urls).
    """
    parse_url = ""
    if parse_urls:
        u = (parse_urls[page_index % len(parse_urls)] or "").rstrip("/")
        if u.endswith("/chat/completions"):
            u = u[: -len("/chat/completions")].rstrip("/")
        parse_url = u
    if not parse_url:
        return (page_num, "", [])
    return render_and_parse_page_impl(
        pdf_path,
        page_index,
        page_num,
        dpi,
        target_size,
        parse_url,
        parse_model,
        parse_max_tokens,
        parse_temperature,
        parse_timeout,
        api_key,
        retry_cfg_dict,
        None,
    )


if RAY_AVAILABLE and ray is not None:
    render_and_parse_page_remote = ray.remote(render_and_parse_page_task)
else:
    render_and_parse_page_remote = None

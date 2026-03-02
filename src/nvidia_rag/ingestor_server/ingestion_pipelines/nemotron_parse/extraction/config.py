# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal config for Nemotron Parse extraction (ingestor-server).

Builds a pipeline config from NvidiaRAGConfig; no embedding/Milvus/Minio.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from nvidia_rag.utils.configuration import NvidiaRAGConfig


def _ns(**kwargs: Any) -> Any:
    """Return a simple namespace object with the given attributes."""
    from types import SimpleNamespace

    return SimpleNamespace(**kwargs)


def build_extraction_config(
    rag_config: NvidiaRAGConfig,
    max_pages_per_pdf: Optional[int] = None,
) -> Any:
    """
    Build extraction pipeline config from NvidiaRAGConfig.

    Uses nemotron_parse.server_url, model_name; pipeline_mode maps to result_mode;
    PDF and retry use defaults. Ray is disabled (sequential path).
    """
    np_cfg = rag_config.nemotron_parse
    # Map pipeline_mode to result_mode: page_as_image -> page_image, page_as_text -> page_elements
    result_mode = "page_image" if (np_cfg.pipeline_mode or "").lower() == "page_as_image" else "page_elements"

    nemotron_parse = _ns(
        url=(np_cfg.server_url or "").strip(),
        urls=None,
        model=np_cfg.model_name or "nvidia/nemotron-parse",
        max_tokens=3500,
        temperature=0,
        timeout=180,
    )
    vlm_url = (getattr(np_cfg, "vlm_server_url", None) or "").strip()
    if vlm_url and vlm_url.endswith("/chat/completions"):
        vlm_url = vlm_url[: -len("/chat/completions")].rstrip("/")
    vlm = _ns(
        url=vlm_url,
        urls=None,
        model="nvidia/nemotron-nano-12b-v2-vl",
        caption=_ns(max_tokens=256, temperature=0.2, top_p=0.9, timeout=120),
        normalization=_ns(max_tokens=1024, temperature=0.1, top_p=0.9, timeout=120),
    )
    prompts = _ns(vlm_caption="Describe this image concisely.", vlm_normalization_relations="")
    use_ray = bool(getattr(np_cfg, "use_ray", False))
    pipeline = _ns(
        result_mode=result_mode,
        use_vlm_caption=bool(getattr(np_cfg, "use_vlm_caption", False)),
        use_vlm_normalization=False,
        max_pages_per_pdf=max_pages_per_pdf,
        bbox_epsilon=0.01,
        ray_max_in_flight_pages=max(1, int(getattr(np_cfg, "ray_max_in_flight_pages", 8))),
    )
    pdf = _ns(dpi=300, target_size=[1536, 2048])
    ray = _ns(use_ray=use_ray, num_cpus=None)
    http_retry = _ns(stop_after_attempt=3, wait_min=2, wait_max=30)
    api_key = getattr(rag_config, "api_key", None) or ""

    return _ns(
        nemotron_parse=nemotron_parse,
        vlm=vlm,
        prompts=prompts,
        pipeline=pipeline,
        pdf=pdf,
        ray=ray,
        http_retry=http_retry,
        api_key=api_key,
    )

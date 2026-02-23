# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Nemotron Parse extraction pipeline for ingestor-server.

Supports sequential or Ray-based parallel page processing and optional VLM captioning.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from PIL import Image as PILImage
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from nvidia_rag.ingestor_server.ingestion_pipelines.nemotron_parse.extraction import (
    pdf_render as pdf_render_mod,
)
from nvidia_rag.ingestor_server.ingestion_pipelines.nemotron_parse.extraction.parse import (
    RAY_AVAILABLE,
    is_retryable_request_error,
    render_and_parse_page_impl,
    render_and_parse_page_remote,
)

logger = logging.getLogger(__name__)

pdfium_get_page_count = pdf_render_mod.pdfium_get_page_count
pdfium_open_document = pdf_render_mod.pdfium_open_document
pdfium_close_document = pdf_render_mod.pdfium_close_document
pdfium_render_page_to_pil = pdf_render_mod.pdfium_render_page_to_pil
encode_image_to_base64 = pdf_render_mod.encode_image_to_base64
ELEMENT_TYPES_WITH_CROP_IMAGE = pdf_render_mod.ELEMENT_TYPES_WITH_CROP_IMAGE
VLM_CAPTION_ELEMENT_TYPES = pdf_render_mod.VLM_CAPTION_ELEMENT_TYPES


def metadata_for_element(
    page_number: int,
    source_file: str,
    bbox: Optional[Dict[str, float]] = None,
    page_id: Optional[int] = None,
    element_id: Optional[str] = None,
    children: Optional[List[str]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"page_number": page_number, "source_file": source_file}
    if bbox is not None:
        meta["bbox"] = bbox
    if page_id is not None:
        meta["page_id"] = page_id
    if element_id is not None:
        meta["element_id"] = element_id
    if children is not None:
        meta["children"] = children
    return meta


class ExtractionPipeline:
    """
    PDF → render → parse → optional VLM caption → result list.
    Supports sequential or Ray-based parallel page processing.
    """

    def __init__(self, config: Any) -> None:
        pdf_render_mod.require_pdfium()
        self.config = config
        self._np = config.nemotron_parse
        self._pl = config.pipeline
        self._pdf = config.pdf
        self._retry = config.http_retry
        self._vlm = config.vlm
        self._parse_url = (self._np.url or "").strip()
        self._parse_urls = [self._parse_url] if self._parse_url else []
        self._vlm_urls: List[str] = []
        if (getattr(config.vlm, "url", None) or "").strip():
            self._vlm_urls = [(config.vlm.url or "").strip()]
        self._vlm_url_index = 0
        self._use_vlm_caption = bool(getattr(self._pl, "use_vlm_caption", False))
        self._use_ray = bool(
            getattr(getattr(config, "ray", None), "use_ray", False) and RAY_AVAILABLE
        )
        self._retry_dict = {
            "stop_after_attempt": getattr(self._retry, "stop_after_attempt", 3),
            "wait_min": getattr(self._retry, "wait_min", 2),
            "wait_max": getattr(self._retry, "wait_max", 30),
        }
        logger.info(
            "ExtractionPipeline: use_ray=%s ray_available=%s use_vlm_caption=%s vlm_urls=%s",
            getattr(getattr(config, "ray", None), "use_ray", False),
            RAY_AVAILABLE,
            self._use_vlm_caption,
            len(self._vlm_urls),
        )

    def _pdf_page_to_image(self, pdf_path: str, page_index: int) -> PILImage.Image:
        ts = tuple(self._pdf.target_size) if self._pdf.target_size else (1536, 2048)
        return pdfium_render_page_to_pil(pdf_path, page_index, self._pdf.dpi, ts)

    @staticmethod
    def _clamp_bbox_coord(
        value: float, low: float = 0.0, high: float = 1.0, epsilon: float = 0.01
    ) -> float:
        value = float(value)
        if value < low + epsilon:
            return low
        if value > high - epsilon:
            return high
        return value

    def _validate_bbox(
        self, bbox: Any, epsilon: Optional[float] = None
    ) -> Optional[Dict[str, float]]:
        if not bbox or not isinstance(bbox, dict):
            return None
        eps = epsilon if epsilon is not None else getattr(self._pl, "bbox_epsilon", 0.01)
        try:
            xmin = self._clamp_bbox_coord(
                bbox.get("xmin", bbox.get("left", 0)), epsilon=eps
            )
            ymin = self._clamp_bbox_coord(
                bbox.get("ymin", bbox.get("top", 0)), epsilon=eps
            )
            xmax = self._clamp_bbox_coord(
                bbox.get("xmax", bbox.get("right", 0)), epsilon=eps
            )
            ymax = self._clamp_bbox_coord(
                bbox.get("ymax", bbox.get("bottom", 0)), epsilon=eps
            )
            if xmax <= xmin or ymax <= ymin:
                return None
            return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        except Exception:
            return None

    @staticmethod
    def _bbox_to_pixels(
        bbox: Dict[str, float], width: int, height: int
    ) -> Tuple[int, int, int, int]:
        xmin = bbox.get("xmin", bbox.get("left", 0))
        ymin = bbox.get("ymin", bbox.get("top", 0))
        xmax = bbox.get("xmax", bbox.get("right", 0))
        ymax = bbox.get("ymax", bbox.get("bottom", 0))

        def _is_norm(v: float) -> bool:
            try:
                return -0.05 <= float(v) <= 1.05
            except Exception:
                return False

        if all(_is_norm(v) for v in (xmin, ymin, xmax, ymax)):
            x1 = int(max(0.0, min(1.0, float(xmin))) * width)
            y1 = int(max(0.0, min(1.0, float(ymin))) * height)
            x2 = int(max(0.0, min(1.0, float(xmax))) * width)
            y2 = int(max(0.0, min(1.0, float(ymax))) * height)
            return x1, y1, x2, y2
        return (
            int(max(0, min(float(xmin), width))),
            int(max(0, min(float(ymin), height))),
            int(max(0, min(float(xmax), width))),
            int(max(0, min(float(ymax), height))),
        )

    @staticmethod
    def _is_degenerate_bbox(bbox: Optional[Dict[str, float]]) -> bool:
        if not bbox:
            return True
        return (bbox.get("xmax", 0) <= bbox.get("xmin", 0)) or (
            bbox.get("ymax", 0) <= bbox.get("ymin", 0)
        )

    def _crop_element_from_page(
        self, page_image: PILImage.Image, bbox: Dict[str, float]
    ) -> Optional[PILImage.Image]:
        if not bbox:
            return None
        w, h = page_image.size
        x1, y1, x2, y2 = self._bbox_to_pixels(bbox, w, h)
        if x2 <= x1 or y2 <= y1:
            return None
        return page_image.crop((x1, y1, x2, y2))

    def _get_next_vlm_url(self) -> str:
        if not self._vlm_urls:
            return ""
        url = self._vlm_urls[self._vlm_url_index % len(self._vlm_urls)]
        self._vlm_url_index += 1
        return url

    def _http_post(
        self, url: str, headers: Dict[str, str], payload: Any, timeout: int
    ) -> requests.Response:
        r = self._retry

        @retry(
            stop=stop_after_attempt(getattr(r, "stop_after_attempt", 3)),
            wait=wait_exponential(
                multiplier=1,
                min=getattr(r, "wait_min", 2),
                max=getattr(r, "wait_max", 30),
            ),
            retry=retry_if_exception(is_retryable_request_error),
            reraise=True,
        )
        def _post() -> requests.Response:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp

        return _post()

    def _get_openai_vlm_client(
        self, base_url: str, timeout: Optional[int] = None
    ) -> Any:
        if OpenAI is None:
            return None
        base_url = (base_url or "").rstrip("/")
        if not base_url:
            return None
        api_key = (getattr(self.config, "api_key", None) or "").strip() or "dummy"
        kwargs: Dict[str, Any] = {"base_url": base_url, "api_key": api_key}
        if timeout is not None:
            kwargs["timeout"] = float(timeout)
        return OpenAI(**kwargs)

    def vlm_caption_element(
        self, page_image: PILImage.Image, element: Dict[str, Any]
    ) -> Optional[str]:
        vlm_url = self._get_next_vlm_url()
        if not vlm_url:
            return None
        elem_type = (element.get("type") or "").strip()
        if elem_type not in VLM_CAPTION_ELEMENT_TYPES:
            return None
        if (element.get("text") or "").strip():
            return None
        bbox = element.get("bbox")
        if not bbox or self._is_degenerate_bbox(bbox):
            return None
        cropped = self._crop_element_from_page(page_image, bbox)
        if cropped is None:
            return None
        b64 = encode_image_to_base64(cropped, fmt="PNG")
        cap = getattr(self._vlm, "caption", None)
        if not cap:
            return None
        prompt = (
            (getattr(getattr(self.config, "prompts", None), "vlm_caption", None) or "")
            .strip()
            or "Describe this image concisely."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            client = self._get_openai_vlm_client(vlm_url, timeout=getattr(cap, "timeout", 120))
            if client is not None:
                r = self._retry

                @retry(
                    stop=stop_after_attempt(getattr(r, "stop_after_attempt", 3)),
                    wait=wait_exponential(
                        multiplier=1,
                        min=getattr(r, "wait_min", 2),
                        max=getattr(r, "wait_max", 30),
                    ),
                    retry=retry_if_exception(is_retryable_request_error),
                    reraise=True,
                )
                def _create() -> Any:
                    return client.chat.completions.create(
                        model=getattr(self._vlm, "model", "nvidia/nemotron-nano-12b-v2-vl"),
                        messages=messages,
                        max_tokens=getattr(cap, "max_tokens", 256),
                        temperature=getattr(cap, "temperature", 0.2),
                        top_p=getattr(cap, "top_p", 0.9),
                    )

                resp = _create()
                content = ""
                if resp.choices:
                    msg = getattr(resp.choices[0], "message", None)
                    content = (getattr(msg, "content", None) or "") or ""
                if isinstance(content, str) and content.strip():
                    return content.strip()
                return None
            endpoint = f"{vlm_url.rstrip('/')}/chat/completions"
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            if getattr(self.config, "api_key", None):
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            payload = {
                "model": getattr(self._vlm, "model", "nvidia/nemotron-nano-12b-v2-vl"),
                "messages": messages,
                "max_tokens": getattr(cap, "max_tokens", 256),
                "temperature": getattr(cap, "temperature", 0.2),
                "top_p": getattr(cap, "top_p", 0.9),
            }
            resp = self._http_post(endpoint, headers, payload, timeout=getattr(cap, "timeout", 120))
            resp.raise_for_status()
            content = (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if isinstance(content, str) and content.strip():
                return content.strip()
        except Exception as e:
            logger.warning("VLM caption failed: %s", e)
        return None

    def _process_single_page_to_results(
        self,
        pdf_path: str,
        page_idx: int,
        page_num: int,
        pdf_stem: str,
        parse_elements: List[Dict[str, Any]],
        page_image_b64: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not parse_elements:
            return []
        elements = [dict(e) for e in parse_elements]
        if (
            page_image_b64
            and (page_image_b64 if isinstance(page_image_b64, str) else "").strip()
        ):
            raw = base64.b64decode(page_image_b64)
            image = PILImage.open(io.BytesIO(raw)).convert("RGB")
            page_b64 = page_image_b64
        else:
            image = self._pdf_page_to_image(pdf_path, page_idx)
            page_b64 = encode_image_to_base64(image)

        result_mode = getattr(self._pl, "result_mode", "page_elements")
        if (
            self._use_vlm_caption
            and self._vlm_urls
            and result_mode == "page_elements"
        ):
            for elem in elements:
                if (elem.get("type") or "").strip() in VLM_CAPTION_ELEMENT_TYPES and not (
                    (elem.get("text") or "").strip()
                ):
                    cap = self.vlm_caption_element(image, elem)
                    if cap:
                        elem["text"] = cap

        source_file = Path(pdf_path).name
        epsilon = getattr(self._pl, "bbox_epsilon", 0.01)
        out: List[Dict[str, Any]] = []

        if result_mode == "page_image":
            layout_parts = [(e.get("text") or "").strip() for e in elements]
            layout_text = "\n\n".join(p for p in layout_parts if p)
            out.append(
                {
                    "text": layout_text,
                    "type": "page",
                    "image": page_b64,
                    "metadata": metadata_for_element(
                        page_num, source_file, bbox=None, page_id=page_num
                    ),
                }
            )
        else:
            for elem in elements:
                text = (elem.get("text") or "").strip() or "(no text)"
                elem_type = elem.get("type", "Unknown")
                bbox = elem.get("bbox")
                bbox = self._validate_bbox(bbox, epsilon=epsilon) if bbox else None
                if elem_type in ELEMENT_TYPES_WITH_CROP_IMAGE and bbox:
                    cropped = self._crop_element_from_page(image, bbox)
                    img_b64 = (
                        encode_image_to_base64(cropped) if cropped else page_b64
                    )
                else:
                    img_b64 = page_b64
                out.append(
                    {
                        "text": text,
                        "type": elem_type,
                        "image": img_b64,
                        "metadata": metadata_for_element(
                            page_num,
                            source_file,
                            bbox=bbox,
                            page_id=elem.get("page_id"),
                            element_id=elem.get("id"),
                            children=elem.get("children"),
                        ),
                    }
                )
        return out

    def process_pdf_sequential(
        self, pdf_path: str
    ) -> Generator[Tuple[int, List[Dict[str, Any]]], None, None]:
        """
        Yield (page_num, result_list) for each page. Sequential; no Ray.
        """
        pdf_path = str(pdf_path)
        if not self._parse_url:
            logger.warning("Nemotron Parse URL not set; skipping extraction.")
            return

        num_pages = pdfium_get_page_count(pdf_path)
        max_pages = getattr(self._pl, "max_pages_per_pdf", None)
        if max_pages is not None:
            num_pages = min(num_pages, max_pages)
        pdf_stem = Path(pdf_path).stem
        doc, _ = pdfium_open_document(pdf_path)
        try:
            for page_idx in range(num_pages):
                page_num = page_idx + 1
                _, page_b64, elements = render_and_parse_page_impl(
                    pdf_path,
                    page_idx,
                    page_num,
                    self._pdf.dpi,
                    tuple(self._pdf.target_size or (1536, 2048)),
                    self._parse_url,
                    self._np.model,
                    int(self._np.max_tokens),
                    float(self._np.temperature),
                    int(self._np.timeout),
                    getattr(self.config, "api_key", "") or "",
                    self._retry_dict,
                    doc,
                )
                results = self._process_single_page_to_results(
                    pdf_path,
                    page_idx,
                    page_num,
                    pdf_stem,
                    elements,
                    page_image_b64=page_b64 or None,
                )
                yield (page_num, results)
        finally:
            pdfium_close_document(doc)

    def process_pdf_pages_generator(
        self, pdf_path: str
    ) -> Generator[Tuple[int, List[Dict[str, Any]]], None, None]:
        """
        Yield (page_num, result_list) for each page using Ray remote tasks.
        Requires config.ray.use_ray=True and Ray installed.
        """
        pdf_path = str(pdf_path)
        if not self._parse_urls:
            logger.warning("Nemotron Parse URL not set; skipping extraction.")
            return
        if not self._use_ray or render_and_parse_page_remote is None:
            raise RuntimeError(
                "process_pdf_pages_generator requires Ray. Set NEMOTRON_PARSE_USE_RAY=true and install ray."
            )
        ts = tuple(self._pdf.target_size) if self._pdf.target_size else (1536, 2048)
        num_pages = pdfium_get_page_count(pdf_path)
        max_pages = getattr(self._pl, "max_pages_per_pdf", None)
        if max_pages is not None:
            num_pages = min(num_pages, max_pages)
        pdf_stem = Path(pdf_path).stem
        max_in_flight = max(1, int(getattr(self._pl, "ray_max_in_flight_pages", 8)))
        ref_to_page_num: Dict[Any, int] = {}
        cache: Dict[int, Tuple[str, List[Dict[str, Any]]]] = {}
        next_to_yield = 1
        submit_idx = 0

        def _submit(page_idx: int) -> None:
            page_num = page_idx + 1
            ref = render_and_parse_page_remote.remote(
                pdf_path,
                page_idx,
                page_num,
                int(self._pdf.dpi),
                ts,
                self._parse_urls,
                self._np.model,
                int(self._np.max_tokens),
                float(self._np.temperature),
                int(self._np.timeout),
                getattr(self.config, "api_key", "") or "",
                self._retry_dict,
            )
            ref_to_page_num[ref] = page_num

        import ray

        while submit_idx < num_pages and len(ref_to_page_num) < max_in_flight:
            _submit(submit_idx)
            submit_idx += 1

        while next_to_yield <= num_pages:
            if next_to_yield in cache:
                page_b64, elements = cache.pop(next_to_yield)
                page_image_b64_arg = (
                    page_b64
                    if (page_b64 and (page_b64.strip() if isinstance(page_b64, str) else True))
                    else None
                )
                logger.info("  Page %s/%s", next_to_yield, num_pages)
                results = self._process_single_page_to_results(
                    pdf_path,
                    next_to_yield - 1,
                    next_to_yield,
                    pdf_stem,
                    elements,
                    page_image_b64=page_image_b64_arg,
                )
                yield (next_to_yield, results)
                next_to_yield += 1
                while submit_idx < num_pages and len(ref_to_page_num) < max_in_flight:
                    _submit(submit_idx)
                    submit_idx += 1
                continue

            done_refs, _ = ray.wait(list(ref_to_page_num.keys()), num_returns=1)
            done_ref = done_refs[0]
            finished_page_num = ref_to_page_num.pop(done_ref, None)
            if finished_page_num is None:
                ray.get(done_ref)
                continue
            try:
                pn, page_b64, elements = ray.get(done_ref)
            except Exception:
                pn = finished_page_num
                page_b64, elements = "", []
            cache[int(pn)] = (page_b64, elements)

    def process_pdf_generator(
        self, pdf_path: str
    ) -> Generator[Tuple[int, List[Dict[str, Any]]], None, None]:
        """
        Yield (page_num, result_list) for each page. Uses Ray when enabled and available, else sequential.
        """
        if self._use_ray and render_and_parse_page_remote is not None:
            yield from self.process_pdf_pages_generator(pdf_path)
        else:
            yield from self.process_pdf_sequential(pdf_path)

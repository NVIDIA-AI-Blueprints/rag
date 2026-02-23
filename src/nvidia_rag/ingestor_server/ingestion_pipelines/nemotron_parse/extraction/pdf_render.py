# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PDF rendering via PDFium (pypdfium2 raw API).

Vendored from nemotron-parse-extraction for ingestor-server extraction pipeline.
"""

from __future__ import annotations

import base64
import ctypes
import io
import math
from typing import Any, Tuple

from PIL import Image as PILImage

try:
    import pypdfium2.raw as pdfium
except ImportError:
    pdfium = None


NEMOTRON_PARSE_OUTPUT_CLASSES = (
    "Text",
    "Title",
    "Section-header",
    "List-item",
    "TOC",
    "Bibliography",
    "Footnote",
    "Page-header",
    "Page-footer",
    "Picture",
    "Formula",
    "Table",
    "Caption",
)

VLM_CAPTION_ELEMENT_TYPES = ("Picture", "Table", "Formula", "Caption")
ELEMENT_TYPES_WITH_CROP_IMAGE = ("Picture", "Table", "Formula", "Caption")


def require_pdfium() -> None:
    if pdfium is None:
        raise RuntimeError(
            "pypdfium2 is required (PDFium backend). Please pip install pypdfium2."
        )


def pdfium_get_page_count(pdf_path: str) -> int:
    require_pdfium()
    doc, count = pdfium_open_document(pdf_path)
    pdfium_close_document(doc)
    return count


def pdfium_open_document(pdf_path: str) -> Tuple[Any, int]:
    require_pdfium()
    doc = pdfium.FPDF_LoadDocument((pdf_path + "\x00").encode("utf-8"), None)
    if not doc:
        raise RuntimeError(f"PDFium failed to load document: {pdf_path}")
    count = int(pdfium.FPDF_GetPageCount(doc))
    return (doc, count)


def pdfium_close_document(doc: Any) -> None:
    if doc is None:
        return
    try:
        pdfium.FPDF_CloseDocument(doc)
    except Exception:
        pass


def pdfium_render_page_from_doc(
    doc: Any,
    page_index: int,
    dpi: int,
    target_size: Tuple[int, int],
    pdf_path: str = "",
) -> PILImage.Image:
    require_pdfium()
    page = None
    bitmap = None
    path_for_error = pdf_path or "<document>"
    try:
        page = pdfium.FPDF_LoadPage(doc, int(page_index))
        if not page:
            raise RuntimeError(
                f"PDFium failed to load page {page_index} from {path_for_error}"
            )
        scale = float(dpi) / 72.0
        w_pt = float(pdfium.FPDF_GetPageWidthF(page))
        h_pt = float(pdfium.FPDF_GetPageHeightF(page))
        width = max(1, int(math.ceil(w_pt * scale)))
        height = max(1, int(math.ceil(h_pt * scale)))
        use_alpha = 0
        bitmap = pdfium.FPDFBitmap_Create(width, height, int(use_alpha))
        if not bitmap:
            raise RuntimeError("PDFium failed to create bitmap")
        pdfium.FPDFBitmap_FillRect(bitmap, 0, 0, width, height, 0xFFFFFFFF)
        flags = int(getattr(pdfium, "FPDF_LCD_TEXT", 0)) | int(
            getattr(pdfium, "FPDF_ANNOT", 0)
        )
        pdfium.FPDF_RenderPageBitmap(bitmap, page, 0, 0, width, height, 0, flags)
        first_item = pdfium.FPDFBitmap_GetBuffer(bitmap)
        buf = ctypes.cast(
            first_item, ctypes.POINTER(ctypes.c_ubyte * (width * height * 4))
        )
        rgba = PILImage.frombuffer(
            "RGBA", (width, height), buf.contents, "raw", "BGRA", 0, 1
        )
        rgba = rgba.copy()
        canvas_rgb = PILImage.new("RGB", (width, height), (255, 255, 255))
        canvas_rgb.paste(rgba.convert("RGB"), (0, 0))
        out = PILImage.new("RGB", target_size, (255, 255, 255))
        canvas_rgb.thumbnail(target_size, PILImage.Resampling.LANCZOS)
        x = (target_size[0] - canvas_rgb.width) // 2
        y = (target_size[1] - canvas_rgb.height) // 2
        out.paste(canvas_rgb, (x, y))
        return out
    finally:
        if bitmap is not None:
            try:
                pdfium.FPDFBitmap_Destroy(bitmap)
            except Exception:
                pass
        if page is not None:
            try:
                pdfium.FPDF_ClosePage(page)
            except Exception:
                pass


def pdfium_render_page_to_pil(
    pdf_path: str,
    page_index: int,
    dpi: int,
    target_size: Tuple[int, int],
) -> PILImage.Image:
    doc, _ = pdfium_open_document(pdf_path)
    try:
        return pdfium_render_page_from_doc(doc, page_index, dpi, target_size, pdf_path)
    finally:
        pdfium_close_document(doc)


def encode_image_to_base64(image: PILImage.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

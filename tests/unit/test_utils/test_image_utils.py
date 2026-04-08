"""Unit tests for shared image utility helpers."""

import base64
import io

from PIL import Image as PILImage

from nvidia_rag.utils.image_utils import convert_image_url_to_png_b64


def _create_test_image_b64(color: str = "red") -> str:
    """Create a small in-memory PNG and return its base64 representation."""
    img = PILImage.new("RGB", (32, 32), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_convert_image_url_to_png_b64_data_url():
    """Data URLs should be normalized to PNG base64 output."""
    test_image = _create_test_image_b64()
    data_url = f"data:image/jpeg;base64,{test_image}"

    result = convert_image_url_to_png_b64(data_url)

    assert isinstance(result, str)
    assert result.startswith("iVBOR")


def test_convert_image_url_to_png_b64_invalid_input_returns_original():
    """Invalid image inputs should be returned unchanged."""
    invalid = "data:image/jpeg;invalid,"

    result = convert_image_url_to_png_b64(invalid)

    assert result == invalid

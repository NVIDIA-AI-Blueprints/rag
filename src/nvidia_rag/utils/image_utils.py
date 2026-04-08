"""Image helpers shared across multimodal RAG components."""

import base64
import io
import logging
import re

from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def convert_image_url_to_png_b64(image_url: str) -> str:
    """Convert a data URL or raw base64 image string to PNG base64.

    Args:
        image_url: Image content in ``data:image/...;base64,...`` format or as a
            raw base64-encoded string.

    Returns:
        Base64-encoded PNG bytes. If the input cannot be converted, the original
        value is returned so callers can decide how to handle it.
    """
    try:
        if image_url.startswith("data:image/"):
            match = re.match(r"data:image/[^;]+;base64,(.+)", image_url)
            if match:
                b64_data = match.group(1)
            else:
                logger.warning("Invalid data URL format: %s...", image_url[:100])
                return image_url
        else:
            b64_data = image_url

        image_bytes = base64.b64decode(b64_data)
        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

        with io.BytesIO() as buffer:
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except Exception as e:
        logger.warning("Failed to convert image URL to PNG: %s", e)
        return image_url

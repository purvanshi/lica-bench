"""SVG rendering via CairoSVG and pixel-diff computation.

Provides deterministic SVG-to-image rendering and per-pixel change
detection between image pairs, used by the LOO scoring and attribution
pipelines.

Adapted from ``incremental_path_scorer.py`` and
``hybrid_concept_attribution.py`` in the SVG-Evaluation-Metrics research
codebase.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def render_svg_to_image(
    svg_code: str, width: int = 384, height: int = 384,
) -> Optional[Image.Image]:
    """Render an SVG string to a PIL RGB image.

    Returns *None* on any rendering failure (malformed SVG, missing Cairo
    library, etc.).

    Parameters
    ----------
    svg_code : str
        SVG XML source.
    width, height : int
        Output raster dimensions in pixels.
    """
    try:
        import cairosvg
    except ImportError:
        raise ImportError(
            "cairosvg is required for SVG rendering. "
            'Install with: pip install -e ".[svg-loo-metrics]"'
        )

    bytestring = svg_code.encode("utf-8") if isinstance(svg_code, str) else svg_code
    try:
        png_data = cairosvg.svg2png(
            bytestring=bytestring,
            output_width=width,
            output_height=height,
            background_color="white",
        )
        return Image.open(io.BytesIO(png_data)).convert("RGB")
    except Exception:
        try:
            png_data = cairosvg.svg2png(
                bytestring=bytestring,
                output_width=width,
                output_height=height,
            )
            return Image.open(io.BytesIO(png_data)).convert("RGB")
        except Exception:
            logger.debug("Failed to render SVG (%d bytes)", len(svg_code))
            return None


def compute_pixel_diff(img_a: Image.Image, img_b: Image.Image) -> np.ndarray:
    """Compute per-pixel change magnitude between two RGB images.

    Returns a float array of shape ``(H, W)`` with values in ``[0, 1]``,
    where each pixel holds the maximum channel-wise absolute difference
    normalised by 255.
    """
    a = np.array(img_a, dtype=np.float64)
    b = np.array(img_b, dtype=np.float64)
    return np.abs(a - b).max(axis=2) / 255.0


def count_changed_pixels(
    img_before: Image.Image,
    img_after: Image.Image,
    threshold: int = 12,
) -> tuple[int, float]:
    """Count pixels whose max-channel difference exceeds *threshold*.

    Returns ``(count, fraction)`` where *fraction* is relative to the
    total pixel count.
    """
    b = np.array(img_before, dtype=np.int16)
    a = np.array(img_after, dtype=np.int16)
    diff = np.abs(a - b).max(axis=2)
    changed = int((diff > threshold).sum())
    total = b.shape[0] * b.shape[1]
    return changed, changed / max(total, 1)

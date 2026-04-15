"""Shared metric implementations for design benchmarks."""

from .core import edit_distance, fid, iou, lpips_score, ssim
from .text import normalize_font_name

__all__ = [
    "edit_distance",
    "fid",
    "iou",
    "lpips_score",
    "normalize_font_name",
    "ssim",
]

# SVG LOO metrics — available when svg-loo-metrics extra is installed.
try:
    from .svg_loo import (
        compute_attribution_matrix,
        compute_structural_metrics,
        element_score_summary,
        score_svg_loo,
    )

    __all__ += [
        "compute_attribution_matrix",
        "compute_structural_metrics",
        "element_score_summary",
        "score_svg_loo",
    ]
except ImportError:
    pass

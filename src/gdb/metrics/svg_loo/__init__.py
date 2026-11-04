"""SVG Leave-One-Out (LOO) evaluation metrics.

A three-tier pipeline for evaluating SVG structural quality:

**Tier 1 — Element Scoring** (``score_svg_loo``):
  Remove each element one at a time and measure CLIP similarity change.
  Classifies elements as helpful, harmful, or neutral.

**Tier 2 — Concept Attribution** (``compute_attribution_matrix``):
  Ground semantic concepts spatially via CLIPSeg heatmaps, then compute
  element-concept overlap using LOO pixel-diff masks.

**Tier 3 — Structural Metrics** (``compute_structural_metrics``):
  From the attribution matrix, derive Purity, Coverage, Compactness,
  and Locality — metrics that predict downstream editing success.

Install dependencies::

    pip install -e ".[svg-loo-metrics]"

Reference
---------
Based on the LOO framework described in the SVG-Evaluation-Metrics paper.
See ``docs/svg_loo_provenance.md`` for source file mapping.
"""

from .attribution import (
    AttributionResult,
    CLIPSegHeatmapGenerator,
    StructuralMetrics,
    compute_attribution_matrix,
    compute_structural_metrics,
    extract_concepts_from_image,
    extract_concepts_from_text,
)
from .parsing import SVGElement, build_partial_svg, parse_svg, parse_svg_subpaths, split_path_d
from .rendering import compute_pixel_diff, count_changed_pixels, render_svg_to_image
from .scoring import (
    CLIPImageScorer,
    IncrementalResult,
    IncrementalScore,
    element_score_summary,
    score_svg_loo,
)

__all__ = [
    # Tier 1: Element scoring
    "CLIPImageScorer",
    "IncrementalResult",
    "IncrementalScore",
    "element_score_summary",
    "score_svg_loo",
    # Tier 2: Concept attribution
    "AttributionResult",
    "CLIPSegHeatmapGenerator",
    "compute_attribution_matrix",
    "extract_concepts_from_image",
    "extract_concepts_from_text",
    # Tier 3: Structural metrics
    "StructuralMetrics",
    "compute_structural_metrics",
    # Low-level building blocks
    "SVGElement",
    "build_partial_svg",
    "compute_pixel_diff",
    "count_changed_pixels",
    "parse_svg",
    "parse_svg_subpaths",
    "render_svg_to_image",
    "split_path_d",
]

"""LOO (Leave-One-Out) element scoring for SVGs.

Renders N+1 versions of an SVG (full + each element removed) and
measures CLIP image similarity against a reference to quantify each
element's contribution to visual quality.

Adapted from ``incremental_path_scorer.py`` in the SVG-Evaluation-Metrics
research codebase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PIL import Image

from .parsing import (
    build_partial_svg,
    parse_svg,
    parse_svg_subpaths,
)
from .rendering import count_changed_pixels, render_svg_to_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class IncrementalScore:
    """Score for one LOO step."""

    element_index: int
    element_tag: str
    element_summary: str
    cumulative_score: float
    delta_score: float
    loo_delta: float = 0.0
    changed_pixels: int = 0
    changed_pixel_frac: float = 0.0
    render_success: bool = True


@dataclass
class IncrementalResult:
    """Full LOO result for one SVG."""

    svg_id: str
    num_elements: int
    final_score: float
    scores: List[IncrementalScore] = field(default_factory=list)
    harmful_elements: List[int] = field(default_factory=list)
    helpful_elements: List[int] = field(default_factory=list)
    neutral_elements: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "svg_id": self.svg_id,
            "num_elements": self.num_elements,
            "final_score": round(self.final_score, 6),
            "harmful_count": len(self.harmful_elements),
            "helpful_count": len(self.helpful_elements),
            "neutral_count": len(self.neutral_elements),
        }


# ---------------------------------------------------------------------------
# CLIP scorer
# ---------------------------------------------------------------------------


class CLIPImageScorer:
    """CLIP-based image-to-image cosine similarity scorer.

    Uses ``open_clip`` (pip-installable) with OpenAI ViT-B/32 weights by
    default.  The scorer instance should be created once and reused across
    many SVG evaluations to amortise model loading.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
    ) -> None:
        try:
            import open_clip
            import torch  # noqa: F811
        except ImportError:
            raise ImportError(
                "open_clip and torch are required for CLIP scoring. "
                'Install with: pip install -e ".[svg-loo-metrics]"'
            )

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device,
        )
        self.model.eval()
        self.name = f"clip_{model_name}"
        self._torch = torch

    def score(self, img: Image.Image, reference: Image.Image) -> float:
        """Compute cosine similarity between two images.  Returns float in [-1, 1]."""
        with self._torch.no_grad():
            feat1 = self._encode(img)
            feat2 = self._encode(reference)
            return float((feat1 @ feat2.T).item())

    def _encode(self, img: Image.Image):
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor).float()
        return features / features.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# LOO scoring pipeline
# ---------------------------------------------------------------------------


def score_svg_loo(
    svg_code: str,
    reference_image: Image.Image,
    *,
    render_size: int = 384,
    use_subpaths: bool = False,
    delta_threshold: float = 0.005,
    scorer: Optional[CLIPImageScorer] = None,
    device: str = "cuda",
    svg_id: str = "unknown",
) -> IncrementalResult:
    """Score each SVG element via leave-one-out against a reference image.

    Parameters
    ----------
    svg_code : str
        SVG XML source.
    reference_image : PIL.Image.Image
        Ground-truth or target rendering to compare against.
    render_size : int
        Resolution for rendering (both width and height).
    use_subpaths : bool
        If *True*, split compound paths at M/m boundaries for finer
        granularity.
    delta_threshold : float
        LOO delta magnitude below which an element is classified as
        neutral.  Paper default is 0.005.
    scorer : CLIPImageScorer, optional
        Pre-initialised scorer.  If *None*, one is created internally
        (slower for batch use).
    device : str
        Torch device for the CLIP model (ignored if *scorer* is provided).
    svg_id : str
        Identifier for this SVG in the result.

    Returns
    -------
    IncrementalResult
        Per-element LOO scores plus harmful/helpful/neutral classification.
    """
    if scorer is None:
        scorer = CLIPImageScorer(device=device)

    # Parse SVG
    if use_subpaths:
        root_attribs, structural, elements, nsmap = parse_svg_subpaths(svg_code)
    else:
        root_attribs, structural, elements, nsmap = parse_svg(svg_code)

    if not elements:
        return IncrementalResult(svg_id=svg_id, num_elements=0, final_score=0.0)

    result = IncrementalResult(
        svg_id=svg_id, num_elements=len(elements), final_score=0.0,
    )

    # --- Pass 1: prefix contribution + changed pixels ---
    prev_score = 0.0
    prev_img: Optional[Image.Image] = None

    for step, elem in enumerate(elements):
        prefix = elements[: step + 1]
        partial_svg = build_partial_svg(root_attribs, structural, prefix, nsmap)
        img = render_svg_to_image(partial_svg, width=render_size, height=render_size)

        if img is None:
            result.scores.append(IncrementalScore(
                element_index=elem.index,
                element_tag=elem.tag,
                element_summary=elem.summary,
                cumulative_score=prev_score,
                delta_score=0.0,
                render_success=False,
            ))
            continue

        score = scorer.score(img, reference_image)
        delta = score - prev_score

        changed_px, changed_frac = 0, 0.0
        if prev_img is not None:
            changed_px, changed_frac = count_changed_pixels(prev_img, img)
        else:
            blank = Image.new("RGB", (render_size, render_size), (255, 255, 255))
            changed_px, changed_frac = count_changed_pixels(blank, img)

        result.scores.append(IncrementalScore(
            element_index=elem.index,
            element_tag=elem.tag,
            element_summary=elem.summary,
            cumulative_score=score,
            delta_score=delta,
            changed_pixels=changed_px,
            changed_pixel_frac=changed_frac,
        ))

        prev_score = score
        prev_img = img

    if result.scores:
        result.final_score = result.scores[-1].cumulative_score

    # --- Pass 2: leave-one-out ---
    if len(elements) >= 2:
        full_score = result.final_score
        for step, score_entry in enumerate(result.scores):
            if not score_entry.render_success:
                continue
            loo_elements = elements[:step] + elements[step + 1:]
            loo_svg = build_partial_svg(root_attribs, structural, loo_elements, nsmap)
            loo_img = render_svg_to_image(loo_svg, width=render_size, height=render_size)
            if loo_img is not None:
                loo_score = scorer.score(loo_img, reference_image)
                score_entry.loo_delta = full_score - loo_score
            else:
                score_entry.loo_delta = full_score

    # --- Classify elements ---
    for score_entry in result.scores:
        if score_entry.loo_delta < -delta_threshold:
            result.harmful_elements.append(score_entry.element_index)
        elif score_entry.loo_delta > delta_threshold:
            result.helpful_elements.append(score_entry.element_index)
        else:
            result.neutral_elements.append(score_entry.element_index)

    return result


def element_score_summary(result: IncrementalResult) -> Dict[str, float]:
    """Compute aggregate metrics from LOO scores.

    Returns a dict with keys: ``mean_loo_delta``, ``harmful_frac``,
    ``helpful_frac``, ``neutral_frac``, ``final_score``.
    """
    n = max(result.num_elements, 1)
    deltas = [s.loo_delta for s in result.scores if s.render_success]
    return {
        "mean_loo_delta": sum(deltas) / max(len(deltas), 1),
        "harmful_frac": len(result.harmful_elements) / n,
        "helpful_frac": len(result.helpful_elements) / n,
        "neutral_frac": len(result.neutral_elements) / n,
        "final_score": result.final_score,
    }

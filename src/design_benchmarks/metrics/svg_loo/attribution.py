"""Concept attribution and structural metrics for SVG quality evaluation.

Combines CLIPSeg spatial heatmaps with LOO pixel-diff maps to build an
element-concept attribution matrix, then derives structural quality
metrics (Purity, Coverage, Compactness, Locality) inspired by the DCI
(Disentangled, Complete, Informative) framework.

Adapted from ``hybrid_concept_attribution.py`` and
``editability_score.py`` in the SVG-Evaluation-Metrics research codebase.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .parsing import (
    SVGElement,
    build_partial_svg,
    parse_svg,
    parse_svg_subpaths,
)
from .rendering import compute_pixel_diff, render_svg_to_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concept extraction (lightweight, no VLM)
# ---------------------------------------------------------------------------


def extract_concepts_from_text(description: str, max_concepts: int = 8) -> List[str]:
    """Extract noun-phrase concepts from a text description.

    This is a lightweight regex-based extractor that requires no external
    model.  It pulls out adjective+noun phrases and deduplicates them.
    For higher quality, supply pre-extracted concepts directly.

    Parameters
    ----------
    description : str
        Natural language description of the target graphic.
    max_concepts : int
        Maximum number of concepts to return.
    """
    if not description or not description.strip():
        return []

    text = description.lower().strip()

    # Remove common preamble patterns
    text = re.sub(r"^(a |an |the |create |draw |generate |make )", "", text)

    # Split on common conjunctions and prepositions
    chunks = re.split(r"\b(?:and|with|on|in|of|at|by|for|from|to|that|which|the)\b", text)

    concepts: List[str] = []
    seen: set = set()

    for chunk in chunks:
        chunk = chunk.strip().strip(".,;:!?")
        if not chunk or len(chunk) < 2:
            continue

        # Take the last 1-3 words as a noun phrase
        words = chunk.split()
        if not words:
            continue

        # Try 2-word and 3-word phrases first, fall back to single word
        for n in (3, 2, 1):
            if len(words) >= n:
                phrase = " ".join(words[-n:]).strip()
                # Filter out pure function words
                content_words = [
                    w for w in phrase.split()
                    if w not in {
                        "a", "an", "the", "is", "are", "was", "were", "be",
                        "it", "its", "this", "that", "very", "some", "each",
                    }
                ]
                if content_words and phrase not in seen:
                    seen.add(phrase)
                    concepts.append(phrase)
                    break

    return concepts[:max_concepts]


# ---------------------------------------------------------------------------
# API-based concept extraction (VLM)
# ---------------------------------------------------------------------------

# Tracks total API calls this process to prevent accidental batch use.
_api_call_count: int = 0
_API_CALL_HARD_LIMIT: int = 200


def _pil_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _pil_to_data_url(image: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(_pil_to_png_bytes(image)).decode()


def _pil_to_b64(image: Image.Image) -> str:
    return base64.b64encode(_pil_to_png_bytes(image)).decode()


def _parse_concept_list(text: str, max_items: int = 8) -> List[str]:
    """Parse a JSON array of concept strings from LLM output."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [
                str(c).strip().lower()
                for c in parsed
                if isinstance(c, str) and len(str(c).strip()) > 1
            ][:max_items]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[([^\]]+)\]", text)
    if match:
        items = match.group(1)
        concepts = [
            s.strip().strip("\"'").lower()
            for s in items.split(",")
            if len(s.strip().strip("\"'")) > 1
        ]
        return concepts[:max_items]

    lines = [
        line.strip().strip("-•*").strip().strip("\"'").lower()
        for line in text.split("\n")
        if line.strip()
    ]
    return [line for line in lines if 1 < len(line) < 50][:max_items]


_CONCEPT_PROMPT = (
    "This is a rendered SVG image.{context} "
    "List exactly {max_concepts} spatially distinct PARTS of this image. "
    "Rules:\n"
    "1. Each part must occupy a different spatial region (no overlapping parts).\n"
    "2. Be specific about location and appearance — use 'red left arrow', "
    "'top-right circle', 'central dark shape' rather than vague labels like "
    "'arrows' or 'shapes' or 'background'.\n"
    "3. Never name the whole image as one part. Break it down.\n"
    "4. For scientific/technical diagrams: name individual structures "
    "(e.g., 'cell membrane', 'nucleus', 'mitochondria').\n"
    "5. For icons/logos: name sub-components by position and color "
    "(e.g., 'upper-left red triangle', 'center white circle', 'bottom blue bar').\n"
    'Return ONLY a JSON array of short noun phrases. Example: '
    '["red roof", "gray chimney", "brown front door", "left window", '
    '"right window", "green garden path"]'
)


def extract_concepts_from_image(
    image: Image.Image,
    *,
    description: str = "",
    provider: str = "gemini",
    max_concepts: int = 6,
    model_id: str | None = None,
    api_key: str | None = None,
) -> List[str]:
    """Extract spatially distinct visual concepts from a rendered image via a VLM API.

    Calls a single VLM API request with the rendered SVG image and returns
    a list of concept noun-phrases.  Falls back to
    :func:`extract_concepts_from_text` if the API call fails or returns
    fewer than 2 concepts.

    **Scale protection:** This function maintains a per-process call counter
    and raises ``RuntimeError`` after {_API_CALL_HARD_LIMIT} calls.  It is
    designed for small-scale evaluation runs, not batch pipelines.

    Parameters
    ----------
    image : PIL.Image.Image
        Rendered SVG (RGB).
    description : str
        Optional text description for context.
    provider : str
        API provider: ``"gemini"``, ``"openai"``, or ``"anthropic"``.
    max_concepts : int
        Number of concepts to request from the VLM.
    model_id : str or None
        Model identifier override.  Defaults per provider:
        gemini → ``gemini-2.0-flash``,
        openai → ``gpt-4o-mini``,
        anthropic → ``claude-sonnet-4-20250514``.
    api_key : str or None
        API key override.  Falls back to the standard environment variable
        for each provider (``GOOGLE_API_KEY``, ``OPENAI_API_KEY``,
        ``ANTHROPIC_API_KEY``).

    Returns
    -------
    list of str
        Concept noun-phrases, lowercased.

    Raises
    ------
    RuntimeError
        If the per-process call limit is exceeded (scale protection).
    """
    global _api_call_count  # noqa: PLW0603

    if _api_call_count >= _API_CALL_HARD_LIMIT:
        raise RuntimeError(
            f"API concept extraction call limit reached ({_API_CALL_HARD_LIMIT}). "
            "This function is not intended for large-scale batch use. "
            "Pre-extract concepts and pass them via ground_truth['concepts'] instead."
        )
    _api_call_count += 1

    context = f' The image description is: "{description}".' if description else ""
    prompt = _CONCEPT_PROMPT.format(context=context, max_concepts=max_concepts)

    provider = provider.lower()
    try:
        if provider in ("gemini", "google"):
            response_text = _call_gemini(image, prompt, model_id, api_key)
        elif provider == "openai":
            response_text = _call_openai(image, prompt, model_id, api_key)
        elif provider == "anthropic":
            response_text = _call_anthropic(image, prompt, model_id, api_key)
        else:
            logger.warning("Unknown provider %r, falling back to text extraction", provider)
            return extract_concepts_from_text(description, max_concepts)
    except Exception:
        logger.warning(
            "API concept extraction failed (%s), falling back to text extraction",
            provider,
            exc_info=True,
        )
        return extract_concepts_from_text(description, max_concepts)

    concepts = _parse_concept_list(response_text, max_concepts)

    if len(concepts) < 2 and description:
        concepts = extract_concepts_from_text(description, max_concepts)

    return concepts


# --- Provider-specific helpers ---


def _call_gemini(
    image: Image.Image, prompt: str, model_id: str | None, api_key: str | None,
) -> str:
    import os

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai is required for Gemini concept extraction. "
            'Install with: pip install -e ".[gemini]"'
        )

    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    client = genai.Client(api_key=key)
    mid = model_id or "gemini-2.0-flash"

    response = client.models.generate_content(
        model=mid,
        contents=[image, prompt],
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=512),
    )
    return response.text or ""


def _call_openai(
    image: Image.Image, prompt: str, model_id: str | None, api_key: str | None,
) -> str:
    import os

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai is required for OpenAI concept extraction. "
            'Install with: pip install -e ".[openai]"'
        )

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
    mid = model_id or "gpt-4o-mini"

    response = client.chat.completions.create(
        model=mid,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _pil_to_data_url(image)}},
            ],
        }],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def _call_anthropic(
    image: Image.Image, prompt: str, model_id: str | None, api_key: str | None,
) -> str:
    import os

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic is required for Anthropic concept extraction. "
            'Install with: pip install -e ".[anthropic]"'
        )

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))
    mid = model_id or "claude-sonnet-4-20250514"

    response = client.messages.create(
        model=mid,
        max_tokens=512,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _pil_to_b64(image),
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return "".join(
        block.text for block in response.content if hasattr(block, "text")
    )


# ---------------------------------------------------------------------------
# CLIPSeg heatmap generation
# ---------------------------------------------------------------------------

# Module-level cache for the CLIPSeg model
_clipseg_cache: Dict[str, object] = {}


class CLIPSegHeatmapGenerator:
    """Generate per-concept spatial heatmaps using CLIPSeg.

    The model is loaded lazily on first use and cached for reuse.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for CLIPSeg.
    device : str
        Torch device.
    """

    def __init__(
        self,
        model_name: str = "CIDAS/clipseg-rd64-refined",
        device: str = "cuda",
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, CLIPSegForImageSegmentation
        except ImportError:
            raise ImportError(
                "torch and transformers are required for CLIPSeg. "
                'Install with: pip install -e ".[svg-loo-metrics]"'
            )

        self.device = device
        cache_key = f"{model_name}_{device}"

        if cache_key not in _clipseg_cache:
            logger.info("Loading CLIPSeg model: %s", model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device)
            model.eval()
            _clipseg_cache[cache_key] = {"processor": processor, "model": model}

        self._processor = _clipseg_cache[cache_key]["processor"]
        self._model = _clipseg_cache[cache_key]["model"]
        self._torch = __import__("torch")

    def generate_heatmaps(
        self, image: Image.Image, concepts: List[str],
    ) -> Dict[str, np.ndarray]:
        """Generate soft heatmaps for each concept.

        Parameters
        ----------
        image : PIL.Image.Image
            Rendered SVG (RGB).
        concepts : list of str
            Concept strings to ground spatially.

        Returns
        -------
        dict mapping concept name to ``(H, W)`` float array in ``[0, 1]``.
        """
        inputs = self._processor(
            text=concepts,
            images=[image] * len(concepts),
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        probs = self._torch.sigmoid(outputs.logits).cpu().numpy()

        w, h = image.size
        heatmaps: Dict[str, np.ndarray] = {}
        for i, concept in enumerate(concepts):
            hm = probs[i]  # (352, 352) from CLIPSeg
            hm_img = Image.fromarray((hm * 255).astype(np.uint8))
            hm_resized = hm_img.resize((w, h), Image.BILINEAR)
            heatmaps[concept] = np.array(hm_resized, dtype=np.float64) / 255.0

        return heatmaps


# ---------------------------------------------------------------------------
# Attribution matrix computation
# ---------------------------------------------------------------------------


@dataclass
class AttributionResult:
    """Element-concept attribution for one SVG.

    Attributes
    ----------
    overlap_matrix : np.ndarray
        Raw overlap matrix of shape ``(n_elements, n_concepts)``.
        Entry ``(i, c)`` is ``sum(H_c * D_i)`` where ``H_c`` is the
        CLIPSeg heatmap and ``D_i`` is the LOO pixel-diff mask.
    concepts : list of str
        Concept labels for columns.
    elements : list of SVGElement
        Parsed SVG elements for rows.
    """

    overlap_matrix: np.ndarray
    concepts: List[str]
    elements: List[SVGElement] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "concepts": self.concepts,
            "num_elements": len(self.elements),
            "overlap_matrix": np.round(self.overlap_matrix, 6).tolist(),
        }


def compute_attribution_matrix(
    svg_code: str,
    concepts: List[str],
    *,
    render_size: int = 384,
    use_subpaths: bool = False,
    diff_threshold: float = 0.02,
    heatmap_generator: Optional[CLIPSegHeatmapGenerator] = None,
    device: str = "cuda",
) -> Optional[AttributionResult]:
    """Compute element-concept attribution via CLIPSeg heatmaps + LOO pixel diffs.

    For each element *i* and concept *c*::

        A(i, c) = sum( H_c(x,y) * D_i(x,y) )

    where ``H_c`` is the CLIPSeg soft heatmap and ``D_i`` is the per-pixel
    change when element *i* is removed.

    Parameters
    ----------
    svg_code : str
        SVG XML source.
    concepts : list of str
        Concept labels to ground.
    render_size : int
        Rendering resolution.
    use_subpaths : bool
        Split compound paths at M/m boundaries.
    diff_threshold : float
        Minimum pixel-diff magnitude to count as "changed".
    heatmap_generator : CLIPSegHeatmapGenerator, optional
        Pre-initialised generator.  Created internally if *None*.
    device : str
        Torch device (ignored if *heatmap_generator* is provided).

    Returns
    -------
    AttributionResult or None
        *None* if the SVG has no elements or cannot be rendered.
    """
    if use_subpaths:
        root_attribs, structural, elements, nsmap = parse_svg_subpaths(svg_code)
    else:
        root_attribs, structural, elements, nsmap = parse_svg(svg_code)

    n_elem = len(elements)
    n_concepts = len(concepts)

    if n_elem == 0 or n_concepts == 0:
        return None

    # Render full SVG
    full_svg = build_partial_svg(root_attribs, structural, elements, nsmap)
    full_img = render_svg_to_image(full_svg, width=render_size, height=render_size)
    if full_img is None:
        return None

    # Generate concept heatmaps
    if heatmap_generator is None:
        heatmap_generator = CLIPSegHeatmapGenerator(device=device)
    concept_heatmaps = heatmap_generator.generate_heatmaps(full_img, concepts)

    # Compute per-element pixel-diff and overlap with concept heatmaps
    overlap_matrix = np.zeros((n_elem, n_concepts))

    logger.info("Computing LOO pixel-diffs for %d elements...", n_elem)
    for ei in range(n_elem):
        loo_elements = elements[:ei] + elements[ei + 1:]

        if not loo_elements:
            loo_img = Image.new("RGB", (render_size, render_size), (255, 255, 255))
        else:
            loo_svg = build_partial_svg(root_attribs, structural, loo_elements, nsmap)
            loo_img = render_svg_to_image(loo_svg, width=render_size, height=render_size)
            if loo_img is None:
                loo_img = Image.new("RGB", (render_size, render_size), (255, 255, 255))

        diff = compute_pixel_diff(full_img, loo_img)  # (H, W) [0, 1]

        for ci, concept in enumerate(concepts):
            hm = concept_heatmaps.get(concept)
            if hm is None:
                continue
            # Resize heatmap if dimensions don't match
            if hm.shape != diff.shape:
                hm_img = Image.fromarray((hm * 255).astype(np.uint8))
                hm_img = hm_img.resize((diff.shape[1], diff.shape[0]), Image.BILINEAR)
                hm = np.array(hm_img, dtype=np.float64) / 255.0
            overlap_matrix[ei, ci] = float(np.sum(hm * diff))

        if (ei + 1) % 20 == 0:
            logger.info("  %d/%d elements done", ei + 1, n_elem)

    return AttributionResult(
        overlap_matrix=overlap_matrix,
        concepts=concepts,
        elements=elements,
    )


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------


@dataclass
class StructuralMetrics:
    """DCI-inspired structural quality metrics for an SVG.

    All values are in ``[0, 1]`` unless noted.

    Attributes
    ----------
    purity : float
        Mean element purity — fraction of each element's positive
        contribution going to its primary concept.  1 = each element
        serves exactly one concept.
    coverage : float
        Fraction of concepts with at least one dedicated element
        (soft, saturated at adaptive threshold).
    compactness : float
        Mean normalised Herfindahl index per concept.  1 = each concept
        is represented by a single element.
    locality : float
        Attribution-weighted mean absolute deviation from z-order
        centroid.  1 = concept elements are clustered together in source
        order.
    crosstalk : float
        Mean off-diagonal crosstalk between concept groups.  0 = editing
        one concept's elements does not affect other concepts.
    contiguity : float
        Mean source-order adjacency of concept element groups.
        1 = perfectly consecutive.
    editability : float
        Composite score: ``coverage * purity``.
    """

    purity: float = 0.0
    coverage: float = 0.0
    compactness: float = 0.0
    locality: float = 0.0
    crosstalk: float = 0.0
    contiguity: float = 0.0
    editability: float = 0.0

    def to_dict(self) -> dict:
        return {
            "purity": round(self.purity, 4),
            "coverage": round(self.coverage, 4),
            "compactness": round(self.compactness, 4),
            "locality": round(self.locality, 4),
            "crosstalk": round(self.crosstalk, 4),
            "contiguity": round(self.contiguity, 4),
            "editability": round(self.editability, 4),
        }


# --- Internal metric functions ---


def _element_purity(row: np.ndarray) -> float:
    """Fraction of total positive contribution going to the primary concept."""
    pos = np.clip(row, 0, None)
    total = pos.sum()
    if total <= 0:
        return 1.0  # no contribution → trivially pure
    return float(pos.max() / total)


def _element_entropy(row: np.ndarray) -> float:
    """Normalised Shannon entropy of positive contributions.  [0, 1]."""
    pos = np.clip(row, 0, None)
    total = pos.sum()
    if total <= 0 or len(pos) < 2:
        return 0.0
    p = pos / total
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p)))
    max_entropy = float(np.log(len(row)))
    if max_entropy <= 0:
        return 0.0
    return entropy / max_entropy


def compute_structural_metrics(
    attribution_matrix: np.ndarray,
    concepts: List[str],
    *,
    coverage_threshold: float = 0.0,
    active_threshold: float = 0.005,
) -> StructuralMetrics:
    """Compute structural quality metrics from an attribution matrix.

    Parameters
    ----------
    attribution_matrix : np.ndarray
        Shape ``(n_elements, n_concepts)``.  Can be the raw overlap
        matrix from :func:`compute_attribution_matrix` or a LOO delta
        matrix from :mod:`editability_score`.
    concepts : list of str
        Concept labels (length must match columns).
    coverage_threshold : float
        Minimum LOO delta for a concept to count as "covered".
        ``0`` (default) uses an adaptive threshold.
    active_threshold : float
        Minimum positive contribution for an element to be "active".
    """
    M = np.asarray(attribution_matrix, dtype=np.float64)
    n_elem, n_concepts = M.shape

    if n_concepts == 0 or n_elem == 0:
        return StructuralMetrics()

    # --- Semantic Coverage ---
    per_concept_coverage = np.array([M[:, ci].max() for ci in range(n_concepts)])

    if coverage_threshold <= 0:
        pos_coverages = per_concept_coverage[per_concept_coverage > 0]
        if len(pos_coverages) > 0:
            coverage_threshold = max(float(np.median(pos_coverages)) * 0.5, 0.005)
        else:
            coverage_threshold = 0.01

    soft_coverages = np.clip(per_concept_coverage / coverage_threshold, 0.0, 1.0)
    semantic_coverage = float(np.mean(soft_coverages))

    # --- Purity ---
    active_mask = np.array([M[ei, :].max() > active_threshold for ei in range(n_elem)])
    active_indices = np.where(active_mask)[0]

    purities = [_element_purity(M[ei, :]) for ei in active_indices]
    mean_purity = float(np.mean(purities)) if purities else 1.0

    # --- Concept Groups (hard assignment) ---
    concept_groups: Dict[str, List[int]] = {c: [] for c in concepts}
    for ei in range(n_elem):
        primary_ci = int(np.argmax(M[ei, :]))
        concept_groups[concepts[primary_ci]].append(ei)

    # --- Crosstalk ---
    crosstalk_matrix = np.zeros((n_concepts, n_concepts))
    for ca in range(n_concepts):
        group_a = concept_groups[concepts[ca]]
        if not group_a:
            continue
        own_contrib = sum(max(M[ei, ca], 0) for ei in group_a)
        if own_contrib <= 0:
            continue
        for cb in range(n_concepts):
            cross_contrib = sum(max(M[ei, cb], 0) for ei in group_a)
            crosstalk_matrix[ca, cb] = cross_contrib / own_contrib

    off_diag = []
    for ca in range(n_concepts):
        for cb in range(n_concepts):
            if ca != cb:
                off_diag.append(crosstalk_matrix[ca, cb])
    mean_crosstalk = float(np.mean(off_diag)) if off_diag else 0.0

    # --- Compactness (Herfindahl) ---
    concept_compactness_values = []
    for ci in range(n_concepts):
        col = np.clip(M[:, ci], 0, None)
        col_sum = col.sum()
        if col_sum <= 0 or n_elem < 2:
            concept_compactness_values.append(1.0)
            continue
        shares = col / col_sum
        herfindahl = float(np.sum(shares ** 2))
        n_active_for_concept = int((col > active_threshold).sum())
        if n_active_for_concept < 2:
            concept_compactness_values.append(1.0)
        else:
            min_h = 1.0 / n_active_for_concept
            compactness = (herfindahl - min_h) / (1.0 - min_h) if min_h < 1.0 else 1.0
            concept_compactness_values.append(max(0.0, compactness))
    mean_compactness = float(np.mean(concept_compactness_values))

    # --- Contiguity ---
    concept_contiguity_values = []
    for ci in range(n_concepts):
        group = sorted(concept_groups[concepts[ci]])
        if len(group) <= 1:
            concept_contiguity_values.append(1.0)
            continue
        gaps = [group[k + 1] - group[k] - 1 for k in range(len(group) - 1)]
        mean_gap = float(np.mean(gaps))
        max_gap = max(n_elem - 1, 1)
        contiguity = 1.0 - (mean_gap / max_gap)
        concept_contiguity_values.append(max(0.0, contiguity))
    mean_contiguity = float(np.mean(concept_contiguity_values))

    # --- Locality (EMD-inspired) ---
    concept_locality_values = []
    for ci in range(n_concepts):
        col = np.clip(M[:, ci], 0, None)
        active_idx = np.where(col > active_threshold)[0]
        if len(active_idx) <= 1:
            concept_locality_values.append(1.0)
            continue
        weights = col[active_idx]
        total_w = weights.sum()
        if total_w <= 0:
            concept_locality_values.append(1.0)
            continue
        weights = weights / total_w
        indices = active_idx.astype(float)
        centroid = float(np.average(indices, weights=weights))
        weighted_mad = float(np.average(np.abs(indices - centroid), weights=weights))
        max_mad = (n_elem - 1) / 2.0
        if max_mad <= 0:
            concept_locality_values.append(1.0)
        else:
            concept_locality_values.append(
                max(0.0, min(1.0, 1.0 - weighted_mad / max_mad)),
            )
    mean_locality = float(np.mean(concept_locality_values))

    # --- Composite ---
    editability = semantic_coverage * mean_purity

    return StructuralMetrics(
        purity=mean_purity,
        coverage=semantic_coverage,
        compactness=mean_compactness,
        locality=mean_locality,
        crosstalk=mean_crosstalk,
        contiguity=mean_contiguity,
        editability=editability,
    )

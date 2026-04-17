# SVG LOO Metrics — Provenance & Implementation Log

This document records how the SVG LOO (Leave-One-Out) evaluation metrics
were integrated into lica-bench from the research codebase, including
exact source file mappings and what was intentionally left behind.

## Source Codebase

**Repository path:** `/mnt/haonan-svg-us-5a/svg-projects/ml-platform/pipelines/svg-generation/scripts/analysis/`

**Paper:** `/mnt/haonan-svg-us-5a/projects/SVG-Evaluation-Metrics/template.tex`

**Branch:** `svg-loo-paper`

The research codebase contains ~35 Python scripts (~22,000 lines) implementing
the full LOO framework, experiments, visualisations, and pipeline orchestration
for the SVG-Evaluation-Metrics paper.

## What Was Extracted

### `parsing.py`

| lica-bench function | Source file | Source location |
|---|---|---|
| `SVGElement` dataclass | `incremental_path_scorer.py` | lines 63-70 |
| `_localname()` | `incremental_path_scorer.py` | lines 106-110 |
| `_element_summary()` | `incremental_path_scorer.py` | lines 113-140 |
| `parse_svg()` | `incremental_path_scorer.py` | lines 143-233 |
| `build_partial_svg()` | `incremental_path_scorer.py` | lines 236-264 |
| `split_path_d()` | `subpath_scorer.py` | lines 58-89 |
| `parse_svg_subpaths()` | `subpath_scorer.py` | lines 123-269 |

**Changes from source:**
- Removed `sys.path` hacks and `from incremental_path_scorer import ...`
- Replaced `print()` with `logging`
- Added type annotations and PEP 8 formatting
- Used `from __future__ import annotations` for `X | None` syntax

### `rendering.py`

| lica-bench function | Source file | Source location |
|---|---|---|
| `render_svg_to_image()` | `incremental_path_scorer.py` | lines 271-295 |
| `compute_pixel_diff()` | `hybrid_concept_attribution.py` | lines 557-565 |
| `count_changed_pixels()` | `incremental_path_scorer.py` | lines 472-479 |

**Changes from source:**
- Lazy import of `cairosvg` with clear install instructions
- Consistent use of `np.float64` dtype

### `scoring.py`

| lica-bench function | Source file | Source location |
|---|---|---|
| `IncrementalScore` dataclass | `incremental_path_scorer.py` | lines 74-89 |
| `IncrementalResult` dataclass | `incremental_path_scorer.py` | lines 92-103 |
| `CLIPImageScorer` class | `incremental_path_scorer.py` | lines 302-336 |
| `score_svg_loo()` | `incremental_path_scorer.py` | lines 482-614 |
| `element_score_summary()` | new (aggregation helper) | — |

**Changes from source:**
- **Switched from `clip` (OpenAI, git-install) to `open_clip_torch` (pip-installable)**
  - `clip.load('ViT-B/32')` → `open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')`
  - Same ViT-B/32 weights, same cosine similarity scores
- Removed `DINOv2Scorer`, `SigLIP2Scorer`, `Qwen3VLEmbeddingScorer` (research alternatives)
- Removed `get_scorer()` factory (only CLIP needed for benchmark)
- Simplified `IncrementalScore` — removed `step` field (redundant with list index)
- Simplified `IncrementalResult` — removed `svg_path`, `reference_path`, `scorer_name` (file-level metadata)
- Added `element_score_summary()` convenience function
- LOO element classification now uses paper threshold (0.005) as default, not 0.01

### `attribution.py`

| lica-bench function | Source file | Source location |
|---|---|---|
| `CLIPSegHeatmapGenerator` class | `hybrid_concept_attribution.py` | lines 223-270 |
| `compute_attribution_matrix()` | `hybrid_concept_attribution.py` | lines 568-703 (as `compute_hybrid_attribution`) |
| `AttributionResult` dataclass | `hybrid_concept_attribution.py` | lines 484-554 (as `HybridAttributionResult`) |
| `_element_purity()` | `editability_score.py` | lines 182-191 |
| `_element_entropy()` | `editability_score.py` | lines 162-179 |
| `compute_structural_metrics()` | `editability_score.py` | lines 205-468 (as `compute_editability`) |
| `StructuralMetrics` dataclass | new (distilled from `EditabilityResult`) | — |
| `extract_concepts_from_text()` | new (lightweight NP chunking) | — |
| `extract_concepts_from_image()` | new (API-based, replaces local Qwen3-VL) | — |
| `_parse_concept_list()` | `hybrid_concept_attribution.py` | lines 191-216 |

**Changes from source:**
- **Replaced local VLM concept extraction** (Qwen3-VL-32B, 32B params) **with API-based extraction** supporting Gemini, OpenAI, and Anthropic. Uses the same prompt from the research code. Includes a hard limit of 200 API calls per process to prevent accidental batch deployment.
- **Added `extract_concepts_from_text()`** — simple regex-based noun-phrase extractor as a zero-dependency fallback when no API key is available
- **Removed `FusedConceptSegmenter`** (SAM3 fusion — separate heavy dependency)
- Replaced mutable default arg `_model_cache: dict = {}` with module-level `_clipseg_cache`
- Simplified `HybridAttributionResult` → `AttributionResult` (removed `concept_share_matrix`, `element_purity_matrix`, `concept_coverage_matrix`, `element_areas`, `heatmap_mass`, `heatmap_coverage` — these are derivable from `overlap_matrix`)
- Distilled `EditabilityResult` (15+ fields) → `StructuralMetrics` (7 fields) — kept only the DCI metrics + composite score
- Removed `_gini_coefficient()` (unused in final paper metrics)
- Removed `mean_group_concentration` metric (model-level predictor, not useful per-SVG)

## What Was Left Behind

### Research Scripts (NOT extracted)

| Script | Lines | Why left behind |
|---|---|---|
| `concept_attribution.py` | 2,428 | Older text-based attribution approach, superseded by hybrid |
| `empirical_editability.py` | 735 | Validates metrics via actual SVG edits — validation tool, not a metric |
| `metric_validation.py` | 515 | Statistical validation (bootstrap CIs, Bonferroni) — one-time analysis |
| `clip_edit_validation.py` | ~500 | CLIP-based edit validation alternative — validation tool |
| `logit_analysis.py` | 1,537 | Token-level logit analysis — model-internal diagnostic |
| `logit_extended_analysis.py` | 999 | Extended logit analyses — research only |
| `concept_difficulty_analysis.py` | ~400 | Concept type taxonomy — analysis only |
| `concept_interference.py` | ~300 | Pairwise concept correlation — analysis only |
| `compute_entropy_correlation.py` | ~200 | Token entropy correlation — null result (r=0.007) |
| `sam3_concept_masks.py` | 544 | SAM3 mask generation — requires separate conda env |

### POC / Experimental Scripts

| Script | Lines | Why left behind |
|---|---|---|
| `poc_synthetic_artifact_cleanup.py` | 1,118 | Artifact detection POC — experimental |
| `poc_artifact_baselines.py` | ~500 | 6-method baseline comparison — experimental |
| `poc_element_resampling.py` | 934 | Online rejection resampling — experimental |
| `poc_best_of_n.py` | 701 | Best-of-N selection — experimental |
| `poc_concept_best_of_n.py` | 711 | Concept-aware best-of-N — experimental |

### Visualisation / Dashboard Scripts

| Script | Lines | Why left behind |
|---|---|---|
| `visualize_loo_contribution.py` | 527 | Interactive HTML LOO reveal — visualisation only |
| `generate_editability_dashboard.py` | 539 | Interactive dashboard — visualisation only |
| `generate_paper_figures.py` | 617 | Publication figures — paper only |
| `render_element_visuals.py` | ~200 | Per-element PNG rendering — visualisation only |
| `extract_reveal_frames.py` | ~200 | Animation frame extraction — paper only |

### Pipeline / Comparison / Aggregation Scripts

| Script | Lines | Why left behind |
|---|---|---|
| `generate_svgs_vllm.py` | 7,380 | SVG generation with vLLM — data generation, not evaluation |
| `aggregate_editability_comparison.py` | 9,433 | Cross-model aggregation — batch analysis |
| `compare_models.py` | ~400 | Cross-model comparison plots — analysis only |
| `compare_concept_attribution.py` | 551 | GT vs model attribution comparison — analysis only |
| `compare_scorers_attribution.py` | ~400 | Scorer comparison — research only |
| `summarize_artifact_cleanup.py` | 924 | Artifact study aggregation — research only |
| `score_model_outputs.py` | ~300 | End-to-end scoring wrapper — pipeline glue |
| `prepare_model_manifests.py` | ~200 | Input directory setup — pipeline glue |
| `run_hybrid_editability_phases.py` | ~300 | Phase orchestration — pipeline glue |

### Alternative Scorers (NOT extracted)

| Scorer | Source location | Why left behind |
|---|---|---|
| `DINOv2Scorer` | `incremental_path_scorer.py:339-395` | Research alternative — not used in paper's final metrics |
| `SigLIP2Scorer` | `incremental_path_scorer.py:398-441` | Research alternative |
| `Qwen3VLEmbeddingScorer` | `incremental_path_scorer.py:444-458` | Research alternative |
| `CLIPTextImageScorer` | `concept_attribution.py` | Older text-scoring approach |
| `SigLIPScorer` | `concept_attribution.py` | Older text-scoring approach |
| `OpenCLIPScorer` | `concept_attribution.py` | Older text-scoring approach |

## Key Design Decisions

1. **`open_clip_torch` over OpenAI `clip`**: The research code uses `pip install git+https://github.com/openai/CLIP.git` which is not pip-installable from PyPI. We switched to `open_clip_torch` which provides the same ViT-B/32 weights and is pip-installable.

2. **API-based VLM concept extraction replaces local model.** The research code uses Qwen3-VL-32B locally (32B params, heavy GPU footprint). The benchmark version uses lightweight API calls to Gemini/OpenAI/Anthropic instead — same prompt, no local GPU memory. A per-process hard limit of 200 calls prevents accidental batch use. Three-tier fallback: pre-extracted concepts in ground truth > API VLM extraction > regex text extraction.

3. **No SAM3**: The research code optionally fuses CLIPSeg with SAM3 masks. SAM3 requires a separate conda environment and is not needed for benchmark-grade metrics. CLIPSeg alone provides sufficient spatial grounding.

4. **Simplified dataclasses**: Research dataclasses carry extensive metadata for debugging and visualisation. The benchmark versions keep only what's needed for metric computation and serialisation.

5. **Module-level model caching**: Research code uses mutable default arguments (`_model_cache: dict = {}`) for caching loaded models. The benchmark code uses module-level dicts to avoid this anti-pattern.

## File Size Comparison

| | Research codebase | lica-bench integration |
|---|---|---|
| Files | ~35 scripts | 5 modules |
| Total lines | ~22,000 | ~1,100 |
| Scope | Full pipeline + experiments + visualisation | Core metrics only |

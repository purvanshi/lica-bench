"""Maps benchmark ids to paths under ``<dataset_root>/benchmarks/``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

BENCHMARK_BUNDLE_SUBPATHS: Dict[str, str] = {
    "category-1": "category/CategoryClassification",
    "category-2": "category/UserIntentPrediction",
    "layout-1": "layout/layout2-intention-to-layout-generation",
    "layout-2": "layout/layout-3-partial-layout-completion",
    "layout-3": "layout/layout4-multi-aspect-ratio",
    "layout-4": "layout/AspectRatioClassification",
    "layout-5": "layout/ComponentCount",
    "layout-6": "layout/ComponentClassification",
    "layout-7": "layout/ComponentDetection",
    "layout-8": "image/image-9-10-Layer-Aware Inpainting",
    "lottie-1": "lottie",
    "lottie-2": "lottie",
    "svg-1": "svg",
    "svg-2": "svg",
    "svg-3": "svg",
    "svg-4": "svg",
    "svg-5": "svg",
    "svg-6": "svg",
    "svg-7": "svg",
    "svg-8": "svg",
    "template-1": "template",
    "template-2": "template",
    "template-3": "template",
    "template-4": "template",
    "template-5": "template",
    "template-6": "template",
    "temporal-1": "temporal/KeyframeOrdering",
    "temporal-2": "temporal/MotionTypeClassification",
    "temporal-3": "temporal/AnimationPropertyExtraction",
    "temporal-4": "temporal/AnimationParameterGeneration",
    "temporal-5": "temporal/MotionTrajectoryGeneration",
    "temporal-6": "temporal/ShortFormVideoLayoutGeneration",
    "typography-1": "typography/FontFamilyClassification",
    "typography-2": "typography/TextColorEstimation",
    "typography-3": "typography/TextParamsEstimation",
    "typography-4": "typography/StyleRanges",
    "typography-5": "typography/CurvedText",
    "typography-6": "typography/TextRotation",
    "typography-7": "typography/Typography-6-Styled-Text-Generation",
    "typography-8": "typography/Typography-6-Styled-Text-Generation",
}


def resolve_benchmark_data_dir(
    benchmark_id: str,
    dataset_root: Union[str, Path],
    *,
    benchmarks_subdir: str = "benchmarks",
) -> Path:
    rel = BENCHMARK_BUNDLE_SUBPATHS.get(benchmark_id)
    if rel is None:
        raise KeyError(
            f"No default data path for {benchmark_id!r}; pass --data or add to "
            "BENCHMARK_BUNDLE_SUBPATHS."
        )
    root = Path(dataset_root).resolve()
    path = root / benchmarks_subdir / rel
    if not path.is_dir():
        raise FileNotFoundError(f"Missing benchmark data directory: {path}")
    return path

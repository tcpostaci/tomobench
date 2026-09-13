"""Evaluation and comparison scaffolding."""

from tomobench.evaluation.comparison import planned_comparison_note
from tomobench.evaluation.eikonal_benchmarks import run_eikonal_benchmarks
from tomobench.evaluation.metrics import PLANNED_METRICS

__all__ = ["PLANNED_METRICS", "planned_comparison_note", "run_eikonal_benchmarks"]

"""Small metric helpers for bounded evaluation workflows."""

from __future__ import annotations

import math
from typing import Sequence

PLANNED_METRICS = ("mae", "rmse")


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Return the mean absolute error over one target vector."""
    _validate_same_length(actual, predicted)
    if not actual:
        raise ValueError("Cannot compute MAE on an empty sequence.")
    return sum(abs(a - b) for a, b in zip(actual, predicted, strict=True)) / len(actual)


def root_mean_squared_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Return the root mean squared error over one target vector."""
    _validate_same_length(actual, predicted)
    if not actual:
        raise ValueError("Cannot compute RMSE on an empty sequence.")
    mean_squared_error = sum((a - b) ** 2 for a, b in zip(actual, predicted, strict=True)) / len(
        actual
    )
    return math.sqrt(mean_squared_error)


def _validate_same_length(actual: Sequence[float], predicted: Sequence[float]) -> None:
    if len(actual) != len(predicted):
        raise ValueError("Metric inputs must have identical lengths.")

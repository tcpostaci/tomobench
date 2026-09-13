"""Canonical target identity helpers shared by audits and grouped evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


TARGET_HASH_SCHEMA = "target_velocity_vector_sha256_v1"


def target_vector_sha256(values: Sequence[float]) -> str:
    """Return the SHA-256 identity of a finite target vector.

    The representation is deliberately independent of JSON formatting and
    platform-native floating-point byte order: values are encoded as a
    contiguous little-endian float64 vector with an explicit schema and length
    header.
    """
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Target identity requires a non-empty one-dimensional vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Target identity requires finite vector values.")
    contiguous = np.ascontiguousarray(array, dtype="<f8")
    header = (
        TARGET_HASH_SCHEMA.encode("ascii")
        + b"\0dtype=<f8\0length="
        + str(contiguous.size).encode("ascii")
        + b"\0"
    )
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()

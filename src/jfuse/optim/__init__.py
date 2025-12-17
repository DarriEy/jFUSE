"""Optimization utilities for jFUSE.

This module provides gradient-based calibration tools using optax optimizers.
"""

from .calibration import (
    Calibrator,
    CalibrationConfig,
    CalibrationState,
    create_optimizer,
    transform_to_bounded,
    transform_to_unbounded,
    clip_to_bounds,
    compute_grad_norm,
    random_search,
)

__all__ = [
    "Calibrator",
    "CalibrationConfig",
    "CalibrationState",
    "create_optimizer",
    "transform_to_bounded",
    "transform_to_unbounded",
    "clip_to_bounds",
    "compute_grad_norm",
    "random_search",
]

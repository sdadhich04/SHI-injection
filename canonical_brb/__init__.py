"""Canonical SHIELD BRB-r health scoring adapted to Project SHIELD datasets."""

from .core import (
    BRBModel, BinaryEventModel, fit_binary_event, fit_brb,
    score_binary_event, score_brb,
)

__all__ = [
    "BRBModel", "BinaryEventModel", "fit_binary_event", "fit_brb",
    "score_binary_event", "score_brb",
]

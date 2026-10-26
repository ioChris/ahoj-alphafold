from __future__ import annotations

import math

import numpy as np
import pandas as pd


FULL_LEVEL_ORDER = ["very_low", "low", "moderate", "high", "very_high", "unknown"]
MERGED_LEVEL_ORDER = ["low", "moderate", "high", "unknown"]


def _is_missing(value: object) -> bool:
    return value is None or pd.isna(value)


def unitize(value: object, *, src_min: float = 0.0, src_max: float = 1.0) -> float:
    if _is_missing(value):
        return float("nan")
    rng = float(src_max - src_min) if src_max != src_min else 1.0
    v = (float(value) - src_min) / rng
    if np.isnan(v):
        return float("nan")
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))


def unitize_probability(p: object, *, mode: str = "sigmoid", k: float = 1.64) -> float:
    if _is_missing(p):
        return float("nan")
    p = float(p)
    if mode == "linear":
        return unitize(p, src_min=0.5, src_max=1.0)

    p = min(max(p, 0.5), 1.0 - 1e-12)
    logits = math.log(p / (1.0 - p))
    p_prime = 1.0 / (1.0 + math.exp(-float(k) * logits))
    return unitize(p_prime, src_min=0.5, src_max=1.0)


def verbal_from_unit(u: object) -> str:
    if _is_missing(u):
        return "unknown"
    u = float(u)
    if u <= 0.20:
        return "very_low"
    if u <= 0.30:
        return "low"
    if u <= 0.60:
        return "moderate"
    if u <= 0.85:
        return "high"
    return "very_high"


def merge_verbal_level(label: object) -> str:
    if _is_missing(label):
        return "unknown"
    label = str(label).strip().lower().replace(" ", "_")
    if label in {"very_low", "low"}:
        return "low"
    if label == "moderate":
        return "moderate"
    if label in {"high", "very_high"}:
        return "high"
    return "unknown"


def probability_level(p: object, *, mode: str = "sigmoid", k: float = 1.64, merge: bool = False) -> str:
    label = verbal_from_unit(unitize_probability(p, mode=mode, k=k))
    return merge_verbal_level(label) if merge else label


def support_level(u: object, *, merge: bool = False) -> str:
    label = verbal_from_unit(unitize(u, src_min=0.0, src_max=1.0))
    return merge_verbal_level(label) if merge else label


def pretty_level(label: object) -> str:
    if _is_missing(label):
        return "Unknown"
    return str(label).strip().replace("_", " ").title()

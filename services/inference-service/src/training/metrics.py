"""Метрики качества модели для офлайн-оценки."""

from __future__ import annotations

import math

from sklearn.metrics import roc_auc_score


def calculate_roc_auc(y_true: list[int], y_score: list[float]) -> float:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if len(y_true) < 2:
        raise ValueError("ROC-AUC requires at least two samples.")
    if len(set(y_true)) < 2:
        raise ValueError("ROC-AUC requires at least two target classes.")

    return float(roc_auc_score(y_true, y_score))


def _top_k_count(total_items: int, top_share: float) -> int:
    if total_items < 1:
        raise ValueError("Metrics require at least one sample.")
    if top_share <= 0 or top_share > 1:
        raise ValueError("top_share must be in range (0, 1].")
    return max(1, math.ceil(total_items * top_share))


def calculate_precision_at_top_share(
    y_true: list[int], y_score: list[float], top_share: float = 0.2
) -> float:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")

    top_k = _top_k_count(len(y_true), top_share)
    ranked_pairs = sorted(zip(y_score, y_true), key=lambda pair: pair[0], reverse=True)
    hits_in_top = sum(target for _, target in ranked_pairs[:top_k])
    return float(hits_in_top / top_k)


def calculate_lift_at_top_share(
    y_true: list[int], y_score: list[float], top_share: float = 0.2
) -> float:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")

    baseline_positive_rate = sum(y_true) / len(y_true)
    if baseline_positive_rate == 0:
        raise ValueError("Lift is undefined when y_true contains no positives.")

    precision_at_top = calculate_precision_at_top_share(
        y_true=y_true,
        y_score=y_score,
        top_share=top_share,
    )
    return float(precision_at_top / baseline_positive_rate)

"""Бизнес-правило для определения метки по порогу."""


def to_label(score: float, threshold: float) -> str:
    return "churn" if score >= threshold else "stay"

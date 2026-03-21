import pytest

from src.training.evaluate import evaluate_binary_classifier
from src.training.metrics import (
    calculate_lift_at_top_share,
    calculate_precision_at_top_share,
    calculate_roc_auc,
)


def test_calculate_roc_auc_returns_expected_value() -> None:
    result = calculate_roc_auc(
        y_true=[0, 1, 0, 1],
        y_score=[0.1, 0.9, 0.2, 0.8],
    )

    assert result == 1.0


def test_calculate_roc_auc_rejects_different_lengths() -> None:
    with pytest.raises(ValueError):
        calculate_roc_auc(y_true=[0, 1], y_score=[0.1])


def test_calculate_roc_auc_requires_two_classes() -> None:
    with pytest.raises(ValueError):
        calculate_roc_auc(y_true=[1, 1, 1], y_score=[0.2, 0.4, 0.8])


def test_evaluate_binary_classifier_returns_metrics_dict() -> None:
    metrics = evaluate_binary_classifier(
        y_true=[0, 1, 0, 1],
        y_score=[0.1, 0.8, 0.3, 0.7],
    )

    assert "roc_auc" in metrics
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision_at_top_share"] == 1.0
    assert metrics["lift_at_top_share"] == 2.0


def test_calculate_precision_at_top_share() -> None:
    result = calculate_precision_at_top_share(
        y_true=[0, 1, 0, 1, 1],
        y_score=[0.1, 0.9, 0.2, 0.8, 0.7],
        top_share=0.4,
    )

    assert result == 1.0


def test_calculate_lift_at_top_share() -> None:
    result = calculate_lift_at_top_share(
        y_true=[0, 1, 0, 1, 1],
        y_score=[0.1, 0.9, 0.2, 0.8, 0.7],
        top_share=0.4,
    )

    assert result > 1.0

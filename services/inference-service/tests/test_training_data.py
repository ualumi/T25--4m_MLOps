from __future__ import annotations

import pandas as pd

from src.training.data import (
    TARGET_COLUMN,
    get_feature_columns,
    split_features_target,
)


def test_get_feature_columns_excludes_id_and_target() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [1, 2],
            "feature_a": [0.1, 0.2],
            "feature_b": [5, 6],
            "churn": [0, 1],
        }
    )

    assert get_feature_columns(frame) == ["feature_a", "feature_b"]


def test_split_features_target_returns_expected_parts() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [1, 2],
            "feature_a": [0.1, 0.2],
            "feature_b": [5, 6],
            TARGET_COLUMN: [0, 1],
        }
    )

    user_ids, features, target = split_features_target(frame)

    assert user_ids.tolist() == ["1", "2"]
    assert list(features.columns) == ["feature_a", "feature_b"]
    assert target.tolist() == [0, 1]

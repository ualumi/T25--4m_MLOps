"""Вспомогательные функции для загрузки и подготовки датасетов."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ID_COLUMN = "user_id"
TARGET_COLUMN = "churn"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.columns if column not in {ID_COLUMN, TARGET_COLUMN}
    ]


def split_features_target(
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    feature_columns = get_feature_columns(frame)
    user_ids = frame[ID_COLUMN].astype(str)
    features = frame[feature_columns]
    target = frame[TARGET_COLUMN].astype(int)
    return user_ids, features, target

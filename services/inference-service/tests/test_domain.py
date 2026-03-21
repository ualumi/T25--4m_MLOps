from src.domain.services.churn_label import to_label


def test_to_label_churn() -> None:
    assert to_label(0.9, 0.5) == "churn"


def test_to_label_stay() -> None:
    assert to_label(0.1, 0.5) == "stay"

import pytest

from src.application.use_cases.build_target_segment import BuildTargetSegmentUseCase
from src.application.use_cases.predict_churn import PredictChurnUseCase
from src.domain.entities.prediction_input import PredictionInput
from src.domain.entities.scoring_candidate import ScoringCandidate


class StubScorer:
    def score(self, features: list[float]) -> float:
        return 0.8 if sum(features) > 1 else 0.2


def test_predict_churn_use_case_positive() -> None:
    use_case = PredictChurnUseCase(scorer=StubScorer(), threshold=0.5)
    result = use_case.execute(PredictionInput(features=[1.0, 1.0]))

    assert result.score == 0.8
    assert result.label == "churn"


def test_predict_churn_use_case_negative() -> None:
    use_case = PredictChurnUseCase(scorer=StubScorer(), threshold=0.5)
    result = use_case.execute(PredictionInput(features=[0.2, 0.1]))

    assert result.score == 0.2
    assert result.label == "stay"


def test_build_target_segment_top_share() -> None:
    use_case = BuildTargetSegmentUseCase(scorer=StubScorer())
    result = use_case.execute(
        candidates=[
            ScoringCandidate(user_id="u1", features=[0.1, 0.1]),
            ScoringCandidate(user_id="u2", features=[1.0, 1.0]),
            ScoringCandidate(user_id="u3", features=[0.8, 0.8]),
            ScoringCandidate(user_id="u4", features=[0.1, 0.2]),
            ScoringCandidate(user_id="u5", features=[0.9, 0.9]),
        ],
        top_share=0.2,
    )

    assert result.total_users == 5
    assert len(result.segment) == 1
    assert result.segment[0].user_id in {"u2", "u3", "u5"}
    assert result.segment[0].rank == 1


def test_build_target_segment_invalid_share() -> None:
    use_case = BuildTargetSegmentUseCase(scorer=StubScorer())
    with pytest.raises(ValueError):
        use_case.execute(
            candidates=[ScoringCandidate(user_id="u1", features=[0.1])],
            top_share=0,
        )

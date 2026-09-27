import pandas as pd
import pytest

from news_classification.deep_learning import parse_binary_answer, stratified_sample


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("FAKE", 0), ("fake.", 0), ("REAL", 1), ("Real news", 1), ("unknown", None)],
)
def test_parse_binary_answer(answer: str, expected: int | None) -> None:
    assert parse_binary_answer(answer) == expected


def test_stratified_sample() -> None:
    frame = pd.DataFrame(
        {
            "content": [f"article {index}" for index in range(12)],
            "label": [0] * 6 + [1] * 6,
        }
    )
    sample = stratified_sample(frame, per_class=3, random_state=42)
    assert sample["label"].value_counts().to_dict() == {0: 3, 1: 3}
    assert len(sample) == 6

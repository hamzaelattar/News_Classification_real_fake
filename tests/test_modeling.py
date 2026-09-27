import pandas as pd

from news_classification.modeling import fit_experiment, prediction_contributions


def test_small_training_pipeline() -> None:
    frame = pd.DataFrame(
        {
            "title": ["fake rumor", "verified report"] * 6,
            "content": ["fake rumor clickbait", "verified report official"] * 6,
            "label": [0, 1] * 6,
        }
    )
    train, validation, test = frame.iloc[:8], frame.iloc[8:10], frame.iloc[10:]
    result = fit_experiment(
        "smoke",
        train,
        validation,
        test,
        text_column="content",
        ngram_range=(1, 1),
    )
    assert result.metrics.loc["test", "accuracy"] == 1.0

    prediction, probabilities, explanation = prediction_contributions(
        result.pipeline, "verified official report"
    )
    assert prediction == 1
    assert probabilities.shape == (2,)
    assert not explanation.empty

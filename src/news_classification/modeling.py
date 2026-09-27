"""Entraînement, évaluation et interprétation des modèles linéaires."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


@dataclass(slots=True)
class ExperimentResult:
    name: str
    pipeline: Pipeline
    metrics: pd.DataFrame
    confusion: np.ndarray
    training_seconds: float
    text_column: str
    ngram_range: tuple[int, int]


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calcule un ensemble compact de métriques adaptées aux deux classes."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def build_pipeline(
    ngram_range: tuple[int, int],
    *,
    max_features: int = 20_000,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=ngram_range,
                    min_df=2,
                    max_features=max_features,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=random_state)),
        ]
    )


def fit_experiment(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    text_column: str,
    ngram_range: tuple[int, int],
) -> ExperimentResult:
    """Entraîne un pipeline sans fuite TF-IDF et l'évalue sur les trois ensembles."""
    pipeline = build_pipeline(ngram_range)
    started = perf_counter()
    pipeline.fit(train[text_column], train["label"])
    training_seconds = perf_counter() - started

    rows = {}
    for split_name, split in (("train", train), ("validation", validation), ("test", test)):
        predictions = pipeline.predict(split[text_column])
        probabilities = pipeline.predict_proba(split[text_column])[:, 1]
        rows[split_name] = classification_metrics(split["label"], predictions, probabilities)

    test_predictions = pipeline.predict(test[text_column])
    return ExperimentResult(
        name=name,
        pipeline=pipeline,
        metrics=pd.DataFrame.from_dict(rows, orient="index"),
        confusion=confusion_matrix(test["label"], test_predictions),
        training_seconds=training_seconds,
        text_column=text_column,
        ngram_range=ngram_range,
    )


def prediction_contributions(
    pipeline: Pipeline,
    text: str,
    *,
    top_n: int = 15,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    """Décompose une prédiction linéaire en contributions TF-IDF × coefficient."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    vector = vectorizer.transform([text])
    indices = vector.indices
    contributions = vector.data * classifier.coef_[0, indices]
    explanation = pd.DataFrame(
        {
            "feature": vectorizer.get_feature_names_out()[indices],
            "tfidf": vector.data,
            "coefficient": classifier.coef_[0, indices],
            "contribution": contributions,
        }
    )
    explanation = explanation.reindex(
        explanation["contribution"].abs().sort_values(ascending=False).index
    ).head(top_n)
    prediction = int(classifier.predict(vector)[0])
    probabilities = classifier.predict_proba(vector)[0]
    return prediction, probabilities, explanation.reset_index(drop=True)

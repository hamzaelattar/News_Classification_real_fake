"""Chargement, nettoyage et découpage des données."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

FAKE_LABEL = 0
REAL_LABEL = 1
REQUIRED_COLUMNS = {"title", "text", "subject", "date"}


def _candidate_data_roots(data_dir: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir))

    if env_dir := os.getenv("NEWS_DATA_DIR"):
        candidates.append(Path(env_dir))

    current = Path.cwd().resolve()
    candidates.extend([current, *current.parents])
    candidates.append(Path("/kaggle/input/datasets/clmentbisaillon/fake-and-real-news-dataset"))

    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _csv_path(root: Path, name: str) -> Path | None:
    for path in (root / name / name, root / name):
        if path.is_file():
            return path
    return None


def resolve_dataset_paths(data_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Trouve Fake.csv et True.csv localement, via NEWS_DATA_DIR ou sur Kaggle."""
    for root in _candidate_data_roots(data_dir):
        fake_path = _csv_path(root, "Fake.csv")
        true_path = _csv_path(root, "True.csv")
        if fake_path and true_path:
            return fake_path, true_path

    raise FileNotFoundError(
        "Impossible de trouver Fake.csv et True.csv. "
        "Placez-les dans le projet ou définissez NEWS_DATA_DIR."
    )


def _validate_columns(frame: pd.DataFrame, source: Path) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonnes absentes dans {source}: {sorted(missing)}")


def load_news_data(
    data_dir: str | Path | None = None,
    *,
    shuffle: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """Charge les deux sources et ajoute la cible binaire `label`."""
    fake_path, true_path = resolve_dataset_paths(data_dir)
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    _validate_columns(fake, fake_path)
    _validate_columns(true, true_path)

    fake = fake.assign(label=FAKE_LABEL)
    true = true.assign(label=REAL_LABEL)
    frame = pd.concat([fake, true], ignore_index=True)
    if shuffle:
        frame = frame.sample(frac=1, random_state=random_state)
    return frame.reset_index(drop=True)


def clean_news_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise les champs texte, supprime les doublons et construit `content`."""
    cleaned = frame.copy()
    for column in ("title", "text", "subject", "date"):
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()

    cleaned = cleaned.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)
    cleaned["content"] = (cleaned["title"] + " " + cleaned["text"]).str.strip()
    cleaned["title_words"] = cleaned["title"].str.split().str.len()
    cleaned["text_words"] = cleaned["text"].str.split().str.len()
    return cleaned


def split_news_data(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Crée une seule séparation stratifiée Train/Validation/Test."""
    holdout_size = test_size + validation_size
    train, holdout = train_test_split(
        frame,
        test_size=holdout_size,
        random_state=random_state,
        stratify=frame["label"],
    )
    relative_test_size = test_size / holdout_size
    validation, test = train_test_split(
        holdout,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=holdout["label"],
    )
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))

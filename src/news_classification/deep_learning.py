"""Expériences GPU optionnelles : fine-tuning DeBERTa et zero-shot Qwen."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
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


@dataclass(slots=True)
class FineTuneConfig:
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    dropout: float = 0.10
    label_smoothing: float = 0.05
    max_grad_norm: float = 0.5
    warmup_ratio: float = 0.10
    random_state: int = 42
    checkpoint_path: Path = Path("artifacts/deberta_best.pt")


@dataclass(slots=True)
class FineTuneResult:
    model: Any
    tokenizer: Any
    metrics: pd.Series
    confusion: np.ndarray
    batch_history: pd.DataFrame
    epoch_history: pd.DataFrame
    training_seconds: float
    test_seconds: float
    device: str


@dataclass(slots=True)
class ZeroShotConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_input_tokens: int = 2_048
    max_new_tokens: int = 5
    random_state: int = 42


SYSTEM_PROMPT = """You are a binary news classification system.
Classify the news article as FAKE or REAL.
Return ONLY one word: FAKE or REAL.
Do not explain your answer and do not add punctuation."""


def _optional_imports():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Installez les dépendances avec `uv sync --extra transformers`."
        ) from exc
    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "AutoConfig": AutoConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _news_dataset_class(torch: Any, dataset_base: Any):
    class NewsDataset(dataset_base):
        def __init__(self, frame, tokenizer, max_length):
            self.texts = frame["content"].astype(str).tolist()
            self.labels = frame["label"].astype(int).tolist()
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            tokens = self.tokenizer(
                self.texts[index],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[index], dtype=torch.long),
            }

    return NewsDataset


def _make_loaders(train, validation, test, tokenizer, config, libraries):
    torch = libraries["torch"]
    dataset_class = _news_dataset_class(torch, libraries["Dataset"])
    loader_class = libraries["DataLoader"]
    options = {
        "batch_size": config.batch_size,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    datasets = [
        dataset_class(frame, tokenizer, config.max_length) for frame in (train, validation, test)
    ]
    train_loader = loader_class(datasets[0], shuffle=True, **options)
    validation_loader = loader_class(datasets[1], shuffle=False, **options)
    test_loader = loader_class(datasets[2], shuffle=False, **options)
    return train_loader, validation_loader, test_loader


def _evaluate_sequence_model(model, loader, criterion, device, torch):
    model.eval()
    losses: list[float] = []
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits
            losses.append(criterion(logits, inputs["labels"]).item())
            probs = torch.softmax(logits, dim=1)
            labels.extend(inputs["labels"].cpu().numpy())
            predictions.extend(probs.argmax(dim=1).cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())
    return {
        "loss": float(np.mean(losses)),
        "labels": np.asarray(labels),
        "predictions": np.asarray(predictions),
        "probabilities": np.asarray(probabilities),
    }


def _sequence_metrics(evaluation: dict) -> pd.Series:
    truth = evaluation["labels"]
    predicted = evaluation["predictions"]
    probability = evaluation["probabilities"]
    return pd.Series(
        {
            "loss": evaluation["loss"],
            "accuracy": accuracy_score(truth, predicted),
            "balanced_accuracy": balanced_accuracy_score(truth, predicted),
            "precision_macro": precision_score(truth, predicted, average="macro", zero_division=0),
            "recall_macro": recall_score(truth, predicted, average="macro", zero_division=0),
            "f1_macro": f1_score(truth, predicted, average="macro", zero_division=0),
            "precision_weighted": precision_score(
                truth, predicted, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(truth, predicted, average="weighted", zero_division=0),
            "f1_weighted": f1_score(truth, predicted, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(truth, probability),
            "log_loss": log_loss(truth, probability),
            "mcc": matthews_corrcoef(truth, predicted),
        },
        name="test",
    )


def fine_tune_deberta(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: FineTuneConfig | None = None,
) -> FineTuneResult:
    """Fine-tune DeBERTa avec dropout, label smoothing et sélection sur Validation F1."""
    config = config or FineTuneConfig()
    libraries = _optional_imports()
    torch = libraries["torch"]
    _seed_everything(config.random_state, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = libraries["AutoTokenizer"].from_pretrained(config.model_name)
    model_config = libraries["AutoConfig"].from_pretrained(config.model_name, num_labels=2)
    for attribute in ("hidden_dropout_prob", "attention_probs_dropout_prob", "cls_dropout"):
        setattr(model_config, attribute, config.dropout)
    model = (
        libraries["AutoModelForSequenceClassification"]
        .from_pretrained(
            config.model_name,
            config=model_config,
            dtype=torch.float32,
        )
        .to(device)
    )

    loaders = _make_loaders(train, validation, test, tokenizer, config, libraries)
    train_loader, validation_loader, test_loader = loaders
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )
    total_steps = len(train_loader) * config.epochs
    scheduler = libraries["get_linear_schedule_with_warmup"](
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_f1, best_loss = -np.inf, np.inf
    batch_rows: list[dict] = []
    epoch_rows: list[dict] = []
    started = perf_counter()
    for epoch in range(config.epochs):
        model.train()
        for batch_number, batch in enumerate(train_loader, start=1):
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits
            loss = criterion(logits, inputs["labels"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss non finie au batch {batch_number}.")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            batch_rows.append(
                {
                    "epoch": epoch + 1,
                    "batch": batch_number,
                    "loss": loss.item(),
                    "gradient_norm": float(gradient_norm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        validation_eval = _evaluate_sequence_model(
            model, validation_loader, criterion, device, torch
        )
        validation_metrics = _sequence_metrics(validation_eval)
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_f1_macro": validation_metrics["f1_macro"],
            }
        )
        current_f1 = validation_metrics["f1_macro"]
        current_loss = validation_metrics["loss"]
        if current_f1 > best_f1 or (np.isclose(current_f1, best_f1) and current_loss < best_loss):
            best_f1, best_loss = current_f1, current_loss
            torch.save(model.state_dict(), config.checkpoint_path)

    training_seconds = perf_counter() - started
    best_weights = torch.load(
        config.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_weights)
    test_started = perf_counter()
    test_evaluation = _evaluate_sequence_model(model, test_loader, criterion, device, torch)
    test_seconds = perf_counter() - test_started
    metrics = _sequence_metrics(test_evaluation)
    metrics["training_seconds"] = training_seconds
    metrics["test_seconds"] = test_seconds
    metrics["test_ms_per_article"] = 1_000 * test_seconds / len(test)

    return FineTuneResult(
        model=model,
        tokenizer=tokenizer,
        metrics=metrics,
        confusion=confusion_matrix(test_evaluation["labels"], test_evaluation["predictions"]),
        batch_history=pd.DataFrame(batch_rows),
        epoch_history=pd.DataFrame(epoch_rows),
        training_seconds=training_seconds,
        test_seconds=test_seconds,
        device=str(device),
    )


def parse_binary_answer(answer: str) -> int | None:
    normalized = str(answer).strip().upper()
    if normalized.startswith("FAKE"):
        return 0
    if normalized.startswith("REAL"):
        return 1
    return None


def stratified_sample(
    frame: pd.DataFrame,
    *,
    per_class: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """Prélève le même nombre d'articles Fake et Real."""
    counts = frame["label"].value_counts()
    if any(counts.get(label, 0) < per_class for label in (0, 1)):
        raise ValueError("Nombre d'articles insuffisant pour créer l'échantillon stratifié.")
    parts = [
        frame[frame["label"].eq(label)].sample(per_class, random_state=random_state)
        for label in (0, 1)
    ]
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


class QwenZeroShotClassifier:
    """Classification FAKE/REAL par génération contrainte, sans fine-tuning."""

    def __init__(self, config: ZeroShotConfig | None = None) -> None:
        self.config = config or ZeroShotConfig()
        libraries = _optional_imports()
        self.torch = libraries["torch"]
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.tokenizer = libraries["AutoTokenizer"].from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype = self.torch.float16 if self.torch.cuda.is_available() else self.torch.float32
        self.model = (
            libraries["AutoModelForCausalLM"]
            .from_pretrained(
                self.config.model_name,
                dtype=dtype,
            )
            .to(self.device)
        )
        self.model.eval()

    def classify(self, content: str) -> tuple[int | None, str, float]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Classify the following news article.\n\nARTICLE:\n{content}",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        started = perf_counter()
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        latency = perf_counter() - started
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip().upper()
        return parse_binary_answer(answer), answer, latency


def evaluate_zero_shot(
    classifier: QwenZeroShotClassifier,
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Évalue Qwen et conserve aussi les réponses non interprétables."""
    rows: list[dict] = []
    for index, row in sample.reset_index(drop=True).iterrows():
        try:
            prediction, answer, latency = classifier.classify(row["content"])
            error = None
        except Exception as exc:  # l'erreur reste visible dans le tableau final
            prediction, answer, latency, error = None, "", np.nan, str(exc)
        rows.append(
            {
                "sample_index": index,
                "true_label": int(row["label"]),
                "prediction": prediction,
                "answer": answer,
                "latency_seconds": latency,
                "error": error,
            }
        )

    details = pd.DataFrame(rows)
    valid = details.dropna(subset=["prediction"]).copy()
    if valid.empty:
        raise RuntimeError("Aucune réponse Qwen valide.")
    truth = valid["true_label"].astype(int)
    predicted = valid["prediction"].astype(int)
    metrics = pd.Series(
        {
            "accuracy": accuracy_score(truth, predicted),
            "balanced_accuracy": balanced_accuracy_score(truth, predicted),
            "precision_macro": precision_score(truth, predicted, average="macro", zero_division=0),
            "recall_macro": recall_score(truth, predicted, average="macro", zero_division=0),
            "f1_macro": f1_score(truth, predicted, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(truth, predicted),
            "average_latency_seconds": valid["latency_seconds"].mean(),
            "valid_predictions": len(valid),
            "invalid_predictions": len(details) - len(valid),
        },
        name="zero_shot_test",
    )
    details["correct"] = details["prediction"].eq(details["true_label"])
    return details, metrics, confusion_matrix(truth, predicted, labels=[0, 1])

"""Construit le notebook principal à partir de cellules courtes et ordonnées."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "LLM.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


cells = [
    markdown(
        """
        # Classification d'actualités : Fake ou Real

        Ce notebook construit une expérience reproductible de classification binaire :

        - **0 — Fake**
        - **1 — Real**

        Le protocole déduplique les articles avant de créer une séparation unique
        Train/Validation/Test. Les variables `subject` et `date` sont analysées, mais ne sont pas
        utilisées par les modèles afin de limiter les fuites d'information.
        """
    ),
    markdown("## 1. Configuration et imports"),
    code(
        """
        import os
        from pathlib import Path

        import joblib
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import ConfusionMatrixDisplay

        from news_classification.data import clean_news_data, load_news_data, split_news_data
        from news_classification.modeling import (
            classification_metrics,
            fit_experiment,
            prediction_contributions,
        )
        """
    ),
    code(
        """
        RANDOM_STATE = 42
        ARTIFACT_DIR = Path("artifacts")
        ARTIFACT_DIR.mkdir(exist_ok=True)

        pd.set_option("display.max_colwidth", 120)
        plt.style.use("seaborn-v0_8-whitegrid")
        """
    ),
    markdown("## 2. Chargement et préparation des données"),
    code(
        """
        raw_df = load_news_data(random_state=RANDOM_STATE)
        print(f"Articles bruts : {len(raw_df):,}")
        raw_df.head(3)
        """
    ),
    code(
        """
        quality_before = pd.Series({
            "valeurs manquantes": int(raw_df.isna().sum().sum()),
            "textes vides ou espaces": int(raw_df["text"].fillna("").str.strip().eq("").sum()),
            "doublons title + text": int(raw_df.duplicated(["title", "text"]).sum()),
        })
        quality_before.to_frame("nombre")
        """
    ),
    code(
        """
        df = clean_news_data(raw_df)
        print(f"Articles après déduplication : {len(df):,}")
        print(f"Doublons restants : {df.duplicated(['title', 'text']).sum()}")
        """
    ),
    code(
        """
        train_df, validation_df, test_df = split_news_data(
            df,
            random_state=RANDOM_STATE,
        )

        split_sizes = pd.Series({
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
        })
        split_sizes.to_frame("articles")
        """
    ),
    code(
        """
        class_balance = pd.concat(
            {
                "train": train_df["label"].value_counts(normalize=True),
                "validation": validation_df["label"].value_counts(normalize=True),
                "test": test_df["label"].value_counts(normalize=True),
            },
            axis=1,
        ).sort_index()
        class_balance.index = ["Fake (0)", "Real (1)"]
        class_balance.round(4)
        """
    ),
    markdown("## 3. Exploration et risques de fuite"),
    markdown("### 3.1 Distribution des classes"),
    code(
        """
        class_counts = df["label"].value_counts().sort_index()
        ax = class_counts.plot.bar(figsize=(6, 4), rot=0, title="Distribution des classes")
        ax.set_xticklabels(["Fake (0)", "Real (1)"])
        ax.set_ylabel("Nombre d'articles")
        plt.show()
        """
    ),
    markdown("### 3.2 Longueur des titres et des articles"),
    code(
        """
        length_summary = df.groupby("label")[["title_words", "text_words"]].agg(
            ["mean", "median", "min", "max"]
        )
        length_summary.rename(index={0: "Fake", 1: "Real"}).round(1)
        """
    ),
    markdown("### 3.3 Corrélation entre `subject` et la cible"),
    code(
        """
        subject_distribution = pd.crosstab(df["subject"], df["label"])
        subject_distribution.columns = ["Fake", "Real"]
        subject_distribution.sort_values(["Fake", "Real"], ascending=False)
        """
    ),
    markdown("### 3.4 Signatures éditoriales"),
    code(
        """
        markers = ["reuters", "featured image", "getty images", "twitter.com"]
        marker_rows = []
        for marker in markers:
            present = df["content"].str.contains(marker, case=False, regex=False)
            marker_rows.append({
                "marqueur": marker,
                "Fake (%)": 100 * present[df["label"].eq(0)].mean(),
                "Real (%)": 100 * present[df["label"].eq(1)].mean(),
            })

        pd.DataFrame(marker_rows).set_index("marqueur").round(2)
        """
    ),
    markdown(
        """
        > **Interprétation.** Les signatures de sources sont très corrélées avec les labels. Une
        excellente accuracy sur ce dataset peut donc mesurer la reconnaissance de l'éditeur plutôt
        que la détection générale d'une fausse information.
        """
    ),
    markdown("## 4. Modèles TF-IDF + régression logistique"),
    markdown("### 4.1 Définition des expériences"),
    code(
        """
        experiment_specs = [
            ("Titre — unigrammes", "title", (1, 1)),
            ("Titre — bigrammes", "title", (2, 2)),
            ("Texte — unigrammes", "text", (1, 1)),
            ("Titre + texte — unigrammes", "content", (1, 1)),
            ("Titre + texte — bigrammes", "content", (2, 2)),
        ]
        """
    ),
    markdown("### 4.2 Entraînement sur un split commun"),
    code(
        """
        experiments = {}
        for name, column, ngrams in experiment_specs:
            print(f"Entraînement : {name}")
            experiments[name] = fit_experiment(
                name,
                train_df,
                validation_df,
                test_df,
                text_column=column,
                ngram_range=ngrams,
            )
        """
    ),
    markdown("### 4.3 Comparaison des résultats"),
    code(
        """
        comparison = pd.DataFrame({
            name: {
                "validation_accuracy": result.metrics.loc["validation", "accuracy"],
                "test_accuracy": result.metrics.loc["test", "accuracy"],
                "test_f1_macro": result.metrics.loc["test", "f1_macro"],
                "test_roc_auc": result.metrics.loc["test", "roc_auc"],
                "training_seconds": result.training_seconds,
            }
            for name, result in experiments.items()
        }).T.sort_values("validation_accuracy", ascending=False)

        comparison.round(4)
        """
    ),
    code(
        """
        best_name = comparison.index[0]
        best_result = experiments[best_name]
        print(f"Modèle sélectionné sur la validation : {best_name}")
        best_result.metrics.round(4)
        """
    ),
    markdown("### 4.4 Matrice de confusion finale"),
    code(
        """
        ConfusionMatrixDisplay(
            confusion_matrix=best_result.confusion,
            display_labels=["Fake", "Real"],
        ).plot(cmap="Blues")
        plt.title(f"Matrice de confusion — {best_name}")
        plt.show()
        """
    ),
    markdown("## 5. Interprétation du meilleur modèle"),
    markdown("### 5.1 Termes les plus influents"),
    code(
        """
        vectorizer = best_result.pipeline.named_steps["tfidf"]
        classifier = best_result.pipeline.named_steps["classifier"]
        coefficients = pd.Series(
            classifier.coef_[0],
            index=vectorizer.get_feature_names_out(),
            name="coefficient",
        )

        top_terms = pd.concat({
            "vers Fake": coefficients.nsmallest(15),
            "vers Real": coefficients.nlargest(15),
        })
        top_terms.to_frame()
        """
    ),
    markdown("### 5.2 Explication locale d'une prédiction"),
    code(
        """
        example = (
            "Government announces new policy after meeting. "
            "Officials in Washington said more details would follow."
        )
        prediction, probabilities, explanation = prediction_contributions(
            best_result.pipeline,
            example,
        )

        print("Classe :", "Real" if prediction == 1 else "Fake")
        print(f"P(Fake)={probabilities[0]:.3f} — P(Real)={probabilities[1]:.3f}")
        explanation.round(4)
        """
    ),
    markdown("## 6. Sauvegarde du pipeline"),
    code(
        """
        model_path = ARTIFACT_DIR / "tfidf_logistic_pipeline.joblib"
        joblib.dump(best_result.pipeline, model_path)
        print(f"Pipeline sauvegardé dans : {model_path.resolve()}")
        """
    ),
    markdown("## 7. Expériences GPU optionnelles"),
    markdown(
        """
        Ces expériences sont désactivées par défaut. Elles téléchargent des modèles volumineux et
        demandent idéalement un GPU. Installez d'abord leurs dépendances avec
        `uv sync --extra transformers`.

        Chaque expérience possède son propre drapeau afin d'éviter de lancer accidentellement un
        entraînement de plusieurs heures.
        """
    ),
    code(
        """
        RUN_FROZEN_DEBERTA = os.getenv("RUN_FROZEN_DEBERTA", "0") == "1"
        RUN_DEBERTA_FINETUNING = os.getenv("RUN_DEBERTA_FINETUNING", "0") == "1"
        RUN_QWEN_ZERO_SHOT = os.getenv("RUN_QWEN_ZERO_SHOT", "0") == "1"

        print("DeBERTa gelé       :", RUN_FROZEN_DEBERTA)
        print("Fine-tuning DeBERTa:", RUN_DEBERTA_FINETUNING)
        print("Qwen zero-shot     :", RUN_QWEN_ZERO_SHOT)
        """
    ),
    markdown("### 7.1 DeBERTa gelé : extraction des embeddings"),
    code(
        """
        if RUN_FROZEN_DEBERTA:
            from news_classification.transformer_features import FrozenDebertaEncoder

            encoder = FrozenDebertaEncoder()
            train_embeddings, train_embedding_seconds = encoder.encode(train_df["content"].tolist())
            validation_embeddings, _ = encoder.encode(validation_df["content"].tolist())
            test_embeddings, _ = encoder.encode(test_df["content"].tolist())
        else:
            print("Section ignorée. Définissez RUN_FROZEN_DEBERTA=1 pour l'exécuter.")
        """
    ),
    markdown("#### Classification des embeddings"),
    code(
        """
        if RUN_FROZEN_DEBERTA:
            deberta_classifier = LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)
            deberta_classifier.fit(train_embeddings, train_df["label"])

            predictions = deberta_classifier.predict(test_embeddings)
            probabilities = deberta_classifier.predict_proba(test_embeddings)[:, 1]
            deberta_metrics = classification_metrics(
                test_df["label"],
                predictions,
                probabilities,
            )
            pd.Series(deberta_metrics, name="test").to_frame().round(4)
        """
    ),
    markdown("### 7.2 Fine-tuning complet de DeBERTa"),
    markdown(
        """
        Cette expérience reprend les nouveaux réglages : un epoch, dropout, label smoothing,
        gradient clipping, warmup et sauvegarde du meilleur modèle selon le F1 de validation.
        Elle utilise les données dédupliquées et le split commun du notebook.
        """
    ),
    code(
        """
        if RUN_DEBERTA_FINETUNING:
            from news_classification.deep_learning import FineTuneConfig, fine_tune_deberta

            fine_tune_config = FineTuneConfig(random_state=RANDOM_STATE)
            fine_tune_result = fine_tune_deberta(
                train_df,
                validation_df,
                test_df,
                fine_tune_config,
            )
        else:
            print("Section ignorée. Définissez RUN_DEBERTA_FINETUNING=1 pour l'exécuter.")
        """
    ),
    markdown("#### Résultats du fine-tuning"),
    code(
        """
        if RUN_DEBERTA_FINETUNING:
            print("Device :", fine_tune_result.device)
            display(fine_tune_result.metrics.to_frame().round(4))
            ConfusionMatrixDisplay(
                fine_tune_result.confusion,
                display_labels=["Fake", "Real"],
            ).plot(cmap="Blues")
            plt.title("DeBERTa fine-tuné — Test")
            plt.show()
        """
    ),
    markdown("#### Courbes d'entraînement"),
    code(
        """
        if RUN_DEBERTA_FINETUNING:
            history = fine_tune_result.batch_history
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(history["batch"], history["loss"])
            axes[0].set(title="Loss par batch", xlabel="Batch", ylabel="Loss")
            axes[1].plot(history["batch"], history["learning_rate"])
            axes[1].set(title="Learning rate", xlabel="Batch", ylabel="Learning rate")
            plt.tight_layout()
            plt.show()
        """
    ),
    markdown("### 7.3 Qwen2.5-1.5B-Instruct en zero-shot"),
    markdown(
        """
        Qwen reçoit uniquement une consigne et doit répondre `FAKE` ou `REAL`. Comme dans le nouveau
        notebook, l'évaluation utilise par défaut 50 articles de chaque classe. Cette expérience ne
        réalise aucun fine-tuning et ne nécessite aucune clé API.
        """
    ),
    code(
        """
        if RUN_QWEN_ZERO_SHOT:
            from news_classification.deep_learning import (
                QwenZeroShotClassifier,
                ZeroShotConfig,
                evaluate_zero_shot,
                stratified_sample,
            )

            qwen_sample = stratified_sample(test_df, per_class=50, random_state=RANDOM_STATE)
            qwen_classifier = QwenZeroShotClassifier(ZeroShotConfig(random_state=RANDOM_STATE))
        else:
            print("Section ignorée. Définissez RUN_QWEN_ZERO_SHOT=1 pour l'exécuter.")
        """
    ),
    markdown("#### Évaluation zero-shot"),
    code(
        """
        if RUN_QWEN_ZERO_SHOT:
            qwen_details, qwen_metrics, qwen_confusion = evaluate_zero_shot(
                qwen_classifier,
                qwen_sample,
            )
            display(qwen_metrics.to_frame().round(4))
        """
    ),
    markdown("#### Matrice de confusion et erreurs"),
    code(
        """
        if RUN_QWEN_ZERO_SHOT:
            ConfusionMatrixDisplay(
                qwen_confusion,
                display_labels=["Fake", "Real"],
            ).plot(cmap="Blues")
            plt.title("Qwen zero-shot — échantillon stratifié")
            plt.show()

            qwen_errors = qwen_details[qwen_details["correct"].eq(False)]
            display(qwen_errors.head(20))
        """
    ),
    markdown("## 8. Conclusion"),
    markdown(
        """
        Le pipeline linéaire offre une base rapide, interprétable et facile à déployer. Avant toute
        utilisation réelle, l'évaluation doit être renforcée avec des articles provenant de sources,
        de périodes et de sujets absents de l'entraînement.
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3 (news-classification)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)

nbf.write(notebook, NOTEBOOK_PATH)
print(f"Notebook écrit : {NOTEBOOK_PATH}")

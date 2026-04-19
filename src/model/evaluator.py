"""Standalone local evaluation utilities for churn classification models."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)
UNKNOWN_CATEGORY = "Unknown"


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Paths to persisted evaluation outputs."""

    metrics_path: Path
    predictions_path: Path
    confusion_matrix_path: Path
    feature_importance_path: Path | None


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation outputs returned to the caller."""

    metrics: dict[str, float]
    artifacts: EvaluationArtifacts


class GenericBinaryClassifierEvaluator:
    """Evaluate a saved local binary classifier bundle against a dataset."""

    def __init__(self, model_path: str | Path, data_path: str | Path, report_dir: str | Path) -> None:
        """Initialize evaluator paths."""

        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.report_dir = Path(report_dir)

    def load_model_bundle(self) -> dict[str, Any]:
        """Load and validate the saved model bundle."""

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {self.model_path}")

        with self.model_path.open("rb") as file:
            bundle = pickle.load(file)

        required_keys = {
            "model",
            "training_features",
            "categorical_features",
            "target_column",
            "label_encoders",
        }
        missing_keys = sorted(required_keys - set(bundle.keys()))
        if missing_keys:
            raise ValueError(
                f"Model bundle is missing required keys: {missing_keys}. "
                f"Available keys: {sorted(bundle.keys())}"
            )

        return bundle

    def load_dataset(self) -> pd.DataFrame:
        """Load the evaluation dataset from parquet or CSV."""

        if not self.data_path.exists():
            raise FileNotFoundError(f"Evaluation data file not found: {self.data_path}")

        suffix = self.data_path.suffix.lower()
        LOGGER.info("Loading evaluation dataset from %s", self.data_path)
        if suffix == ".parquet":
            return pd.read_parquet(self.data_path)
        if suffix == ".csv":
            return pd.read_csv(self.data_path)

        raise ValueError(
            f"Unsupported evaluation data format '{suffix}'. Use a .parquet or .csv file."
        )

    @staticmethod
    def validate_dataset(
        df: pd.DataFrame,
        training_features: list[str],
        target_column: str,
    ) -> None:
        """Validate that required evaluation columns exist."""

        required_columns = set(training_features)
        required_columns.add(target_column)
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(
                "Evaluation dataset is missing required columns: "
                f"{missing_columns}. Available columns: {list(df.columns)}"
            )

    @staticmethod
    def _normalize_categorical_series(series: pd.Series) -> pd.Series:
        """Normalize categorical values using the same strategy as training."""

        return (
            series.astype("string")
            .fillna(UNKNOWN_CATEGORY)
            .str.strip()
            .replace("", UNKNOWN_CATEGORY)
        )

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        training_features: list[str],
        categorical_features: list[str],
        label_encoders: dict[str, Any],
    ) -> pd.DataFrame:
        """Rebuild the evaluation feature matrix in exact saved feature order."""

        feature_df = df.loc[:, training_features].copy()

        numeric_features = [
            column for column in training_features if column not in categorical_features
        ]
        for column in numeric_features:
            feature_df[column] = pd.to_numeric(feature_df[column], errors="raise")
            if feature_df[column].isna().any():
                raise ValueError(f"Numeric evaluation feature '{column}' contains missing values.")

        for column in categorical_features:
            if column not in label_encoders:
                raise ValueError(f"Missing label encoder for categorical feature '{column}'.")

            encoder = label_encoders[column]
            known_classes = set(encoder.classes_.tolist())
            if UNKNOWN_CATEGORY not in known_classes:
                raise ValueError(
                    f"Label encoder for '{column}' does not include '{UNKNOWN_CATEGORY}', "
                    "so unseen category handling is unsafe."
                )

            normalized_series = self._normalize_categorical_series(feature_df[column])
            safe_values = normalized_series.where(normalized_series.isin(known_classes), UNKNOWN_CATEGORY)
            feature_df[column] = encoder.transform(safe_values)

        return feature_df.loc[:, training_features].copy()

    @staticmethod
    def build_target(df: pd.DataFrame, target_column: str) -> pd.Series:
        """Validate and return the evaluation target."""

        target = pd.to_numeric(df[target_column], errors="raise")
        if target.isna().any():
            raise ValueError(f"Evaluation target column '{target_column}' contains missing values.")

        unique_values = set(target.astype(int).unique().tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Evaluation target column '{target_column}' must be binary 0/1. "
                f"Found values: {sorted(unique_values)}"
            )
        return target.astype(int)

    @staticmethod
    def compute_metrics(model: Any, X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, float], pd.DataFrame]:
        """Compute evaluation metrics and return a predictions dataframe."""

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y, predictions)),
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "f1": float(f1_score(y, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
        }
        predictions_df = pd.DataFrame(
            {
                "actual": y,
                "predicted": predictions,
                "predicted_probability": probabilities,
            },
            index=X.index,
        )
        return metrics, predictions_df

    def save_metrics(self, metrics: dict[str, float]) -> Path:
        """Save evaluation metrics as JSON."""

        metrics_path = self.report_dir / "evaluation_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        return metrics_path

    def save_predictions(self, predictions_df: pd.DataFrame, source_df: pd.DataFrame) -> Path:
        """Save evaluation predictions as CSV."""

        predictions_path = self.report_dir / "predictions.csv"
        output_df = predictions_df.copy()
        if "customer_id" in source_df.columns:
            output_df.insert(0, "customer_id", source_df.loc[predictions_df.index, "customer_id"].values)
        output_df.to_csv(predictions_path, index=False)
        return predictions_path

    def save_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series) -> Path:
        """Render and save a confusion matrix plot."""

        confusion_matrix_path = self.report_dir / "confusion_matrix.png"
        matrix = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix=matrix).plot(ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(confusion_matrix_path, dpi=150)
        plt.close(fig)
        return confusion_matrix_path

    def save_feature_importance(self, model: Any, training_features: list[str]) -> Path | None:
        """Render and save feature importances when supported by the model."""

        if not hasattr(model, "feature_importances_"):
            LOGGER.info("Loaded model does not expose feature_importances_; skipping plot.")
            return None

        feature_importance_path = self.report_dir / "feature_importance.png"
        importance_df = pd.DataFrame(
            {
                "feature": training_features,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=importance_df, x="importance", y="feature", ax=ax)
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        fig.savefig(feature_importance_path, dpi=150)
        plt.close(fig)
        return feature_importance_path

    def evaluate(self) -> EvaluationResult:
        """Run standalone evaluation using a saved model bundle."""

        self.report_dir.mkdir(parents=True, exist_ok=True)

        bundle = self.load_model_bundle()
        df = self.load_dataset()

        training_features = list(bundle["training_features"])
        categorical_features = list(bundle["categorical_features"])
        target_column = str(bundle["target_column"])
        label_encoders = dict(bundle["label_encoders"])
        model = bundle["model"]

        self.validate_dataset(df, training_features, target_column)
        X = self.build_feature_matrix(
            df=df,
            training_features=training_features,
            categorical_features=categorical_features,
            label_encoders=label_encoders,
        )
        y = self.build_target(df, target_column)

        metrics, predictions_df = self.compute_metrics(model, X, y)
        metrics_path = self.save_metrics(metrics)
        predictions_path = self.save_predictions(predictions_df, df)
        confusion_matrix_path = self.save_confusion_matrix(y, predictions_df["predicted"])
        feature_importance_path = self.save_feature_importance(model, training_features)

        return EvaluationResult(
            metrics=metrics,
            artifacts=EvaluationArtifacts(
                metrics_path=metrics_path,
                predictions_path=predictions_path,
                confusion_matrix_path=confusion_matrix_path,
                feature_importance_path=feature_importance_path,
            ),
        )

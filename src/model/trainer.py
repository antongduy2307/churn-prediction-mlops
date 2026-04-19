"""Local baseline trainer for deterministic customer churn classification."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

LOGGER = logging.getLogger(__name__)
UNKNOWN_CATEGORY = "Unknown"


@dataclass(frozen=True)
class DataConfig:
    """Dataset location configuration."""

    training_data_path: Path


@dataclass(frozen=True)
class FeatureConfig:
    """Feature and target configuration."""

    target_column: str
    training_features: list[str]
    categorical_features: list[str]
    excluded_columns: list[str]


@dataclass(frozen=True)
class SplitConfig:
    """Train/test split settings."""

    test_size: float
    random_state: int
    stratify: bool


@dataclass(frozen=True)
class OutputConfig:
    """Local artifact output paths."""

    model_bundle_path: Path
    metrics_path: Path | None = None


@dataclass(frozen=True)
class MLflowConfig:
    """Local MLflow tracking configuration."""

    tracking_uri: str
    experiment_name: str
    run_name_prefix: str


@dataclass(frozen=True)
class TrainingConfig:
    """Combined training configuration."""

    data: DataConfig
    features: FeatureConfig
    split: SplitConfig
    model_name: str
    model_params: dict[str, Any]
    output: OutputConfig
    mlflow: MLflowConfig | None = None


@dataclass(frozen=True)
class TrainingResult:
    """Training outputs returned to the caller."""

    model: RandomForestClassifier
    metrics: dict[str, float]
    preprocessing_metadata: dict[str, Any]
    artifact_path: Path


def build_training_config(config_dict: dict[str, Any], base_dir: Path) -> TrainingConfig:
    """Build a typed training config from the YAML payload."""

    def resolve_path(path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (base_dir / path).resolve()

    data_section = config_dict["data"]
    feature_section = config_dict["features"]
    split_section = config_dict["split"]
    model_section = config_dict["model"]
    output_section = config_dict["output"]
    mlflow_section = config_dict.get("mlflow")

    return TrainingConfig(
        data=DataConfig(
            training_data_path=resolve_path(data_section["training_data_path"]),
        ),
        features=FeatureConfig(
            target_column=feature_section["target_column"],
            training_features=list(feature_section["training_features"]),
            categorical_features=list(feature_section["categorical_features"]),
            excluded_columns=list(feature_section["excluded_columns"]),
        ),
        split=SplitConfig(
            test_size=float(split_section["test_size"]),
            random_state=int(split_section["random_state"]),
            stratify=bool(split_section["stratify"]),
        ),
        model_name=str(model_section["name"]),
        model_params=dict(model_section.get("params", {})),
        output=OutputConfig(
            model_bundle_path=resolve_path(output_section["model_bundle_path"]),
            metrics_path=resolve_path(output_section["metrics_path"])
            if output_section.get("metrics_path")
            else None,
        ),
        mlflow=MLflowConfig(
            tracking_uri=str(mlflow_section["tracking_uri"]),
            experiment_name=str(mlflow_section["experiment_name"]),
            run_name_prefix=str(mlflow_section["run_name_prefix"]),
        )
        if mlflow_section
        else None,
    )


class GenericBinaryClassifierTrainer:
    """Trainer for a local baseline binary churn classifier."""

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the trainer with validated configuration."""

        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate internal consistency of the training config."""

        duplicate_features = {
            feature
            for feature in self.config.features.training_features
            if self.config.features.training_features.count(feature) > 1
        }
        if duplicate_features:
            raise ValueError(f"Duplicate training features are not allowed: {sorted(duplicate_features)}")

        invalid_categorical = set(self.config.features.categorical_features) - set(
            self.config.features.training_features
        )
        if invalid_categorical:
            raise ValueError(
                "Categorical features must be a subset of training features: "
                f"{sorted(invalid_categorical)}"
            )

        overlap_with_excluded = set(self.config.features.training_features) & set(
            self.config.features.excluded_columns
        )
        if overlap_with_excluded:
            raise ValueError(
                "Excluded columns cannot also be training features: "
                f"{sorted(overlap_with_excluded)}"
            )

        if self.config.features.target_column in self.config.features.training_features:
            raise ValueError("Target column cannot be part of training features.")

    def load_dataset(self) -> pd.DataFrame:
        """Load the training dataset from parquet or CSV."""

        data_path = self.config.data.training_data_path
        if not data_path.exists():
            raise FileNotFoundError(f"Training data file not found: {data_path}")

        suffix = data_path.suffix.lower()
        LOGGER.info("Loading training dataset from %s", data_path)
        if suffix == ".parquet":
            return pd.read_parquet(data_path)
        if suffix == ".csv":
            return pd.read_csv(data_path)

        raise ValueError(
            f"Unsupported training data format '{suffix}'. Use a .parquet or .csv file."
        )

    def validate_dataset(self, df: pd.DataFrame) -> None:
        """Validate that the dataset contains the required training schema."""

        required_columns = set(self.config.features.training_features)
        required_columns.add(self.config.features.target_column)
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(
                "Training dataset is missing required columns: "
                f"{missing_columns}. Available columns: {list(df.columns)}"
            )

        missing_target = int(df[self.config.features.target_column].isna().sum())
        if missing_target:
            raise ValueError(
                f"Target column '{self.config.features.target_column}' contains {missing_target} missing values."
            )

    def _prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Validate and return the binary target series."""

        target = pd.to_numeric(df[self.config.features.target_column], errors="raise")
        if target.isna().any():
            raise ValueError(f"Target column '{self.config.features.target_column}' contains missing values.")

        unique_values = set(target.astype(int).unique().tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Target column '{self.config.features.target_column}' must be binary 0/1. "
                f"Found values: {sorted(unique_values)}"
            )
        return target.astype(int)

    def _prepare_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and validate training feature columns."""

        feature_df = df.loc[:, self.config.features.training_features].copy()
        numeric_features = [
            column
            for column in self.config.features.training_features
            if column not in self.config.features.categorical_features
        ]

        for column in numeric_features:
            feature_df[column] = pd.to_numeric(feature_df[column], errors="raise")
            if feature_df[column].isna().any():
                raise ValueError(f"Numeric feature '{column}' contains missing values.")

        return feature_df

    @staticmethod
    def _normalize_categorical_series(series: pd.Series) -> pd.Series:
        """Normalize categorical values before label encoding."""

        return (
            series.astype("string")
            .fillna(UNKNOWN_CATEGORY)
            .str.strip()
            .replace("", UNKNOWN_CATEGORY)
        )

    def _fit_label_encoders(self, feature_df: pd.DataFrame) -> dict[str, LabelEncoder]:
        """Fit label encoders on the training partition only."""

        encoders: dict[str, LabelEncoder] = {}
        for column in self.config.features.categorical_features:
            encoder = LabelEncoder()
            normalized_series = self._normalize_categorical_series(feature_df[column])
            classes = sorted(set(normalized_series.tolist()) | {UNKNOWN_CATEGORY})
            encoder.fit(classes)
            encoders[column] = encoder
        return encoders

    def _transform_features(
        self,
        feature_df: pd.DataFrame,
        encoders: dict[str, LabelEncoder],
    ) -> pd.DataFrame:
        """Transform categorical columns with fitted encoders and keep column order stable."""

        transformed_df = feature_df.copy()
        for column, encoder in encoders.items():
            normalized_series = self._normalize_categorical_series(transformed_df[column])
            known_classes = set(encoder.classes_.tolist())
            safe_values = normalized_series.where(normalized_series.isin(known_classes), UNKNOWN_CATEGORY)
            transformed_df[column] = encoder.transform(safe_values)

        return transformed_df.loc[:, self.config.features.training_features].copy()

    @staticmethod
    def _compute_metrics(
        model: RandomForestClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Compute baseline binary classification metrics."""

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        return {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

    def _build_bundle(
        self,
        model: RandomForestClassifier,
        metrics: dict[str, float],
        label_encoders: dict[str, LabelEncoder],
    ) -> dict[str, Any]:
        """Build the saved model bundle payload."""

        return {
            "model": model,
            "training_features": list(self.config.features.training_features),
            "categorical_features": list(self.config.features.categorical_features),
            "target_column": self.config.features.target_column,
            "label_encoders": label_encoders,
            "metrics": metrics,
            "model_name": self.config.model_name,
            "model_params": dict(self.config.model_params),
        }

    def save_model_bundle(
        self,
        bundle: dict[str, Any],
        metrics: dict[str, float],
    ) -> Path:
        """Persist the trained model bundle and optional metrics artifact."""

        artifact_path = self.config.output.model_bundle_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("wb") as file:
            pickle.dump(bundle, file)

        if self.config.output.metrics_path is not None:
            metrics_path = self.config.output.metrics_path
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=2)

        return artifact_path

    def run(self) -> TrainingResult:
        """Execute the local baseline training flow."""

        df = self.load_dataset()
        self.validate_dataset(df)

        X = self._prepare_feature_frame(df)
        y = self._prepare_target(df)

        stratify_values = y if self.config.split.stratify else None
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.split.test_size,
            random_state=self.config.split.random_state,
            stratify=stratify_values,
        )

        label_encoders = self._fit_label_encoders(X_train_raw)
        X_train = self._transform_features(X_train_raw, label_encoders)
        X_test = self._transform_features(X_test_raw, label_encoders)

        model = RandomForestClassifier(**self.config.model_params)
        LOGGER.info("Training %s with %s rows", self.config.model_name, len(X_train))
        model.fit(X_train, y_train)

        metrics = self._compute_metrics(model, X_test, y_test)
        bundle = self._build_bundle(model, metrics, label_encoders)
        artifact_path = self.save_model_bundle(bundle, metrics)

        preprocessing_metadata = {
            "training_features": list(self.config.features.training_features),
            "categorical_features": list(self.config.features.categorical_features),
            "target_column": self.config.features.target_column,
            "label_encoder_classes": {
                column: encoder.classes_.tolist()
                for column, encoder in label_encoders.items()
            },
        }

        return TrainingResult(
            model=model,
            metrics=metrics,
            preprocessing_metadata=preprocessing_metadata,
            artifact_path=artifact_path,
        )

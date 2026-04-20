"""CLI entrypoint for local baseline customer churn training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path

import yaml

from src.model.trainer import GenericBinaryClassifierTrainer, build_training_config

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for CLI execution."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train a local RandomForest churn baseline from a YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training YAML config.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict:
    """Load a YAML config file."""

    if not config_path.exists():
        raise FileNotFoundError(f"Training config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a YAML mapping: {config_path}")

    return config


def log_resolved_config_paths(config: dict, project_root: Path) -> None:
    """Log how relative config paths resolve against the project root."""

    path_fields = [
        ("data.training_data_path", config.get("data", {}).get("training_data_path")),
        ("output.model_bundle_path", config.get("output", {}).get("model_bundle_path")),
        ("output.metrics_path", config.get("output", {}).get("metrics_path")),
    ]

    for field_name, raw_path in path_fields:
        if not raw_path:
            continue
        resolved_path = (project_root / Path(raw_path)).resolve()
        LOGGER.debug(
            "Resolved config path %s: original='%s', resolved='%s'",
            field_name,
            raw_path,
            resolved_path,
        )


def resolve_mlflow_tracking_uri(training_config) -> str | None:
    """Resolve MLflow tracking URI with environment override support."""

    if training_config.mlflow is None:
        return None

    return os.getenv("MLFLOW_TRACKING_URI", training_config.mlflow.tracking_uri)


def should_skip_mlflow_registration() -> bool:
    """Return whether MLflow model registration should be skipped."""

    raw_value = os.getenv("SKIP_MLFLOW_REGISTRATION", "")
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def log_training_run_to_mlflow(
    training_config,
    training_result,
    config_snapshot: dict,
) -> tuple[str, str] | None:
    """Log the completed local training run to MLflow."""

    if training_config.mlflow is None:
        LOGGER.info("MLflow config not provided; skipping MLflow logging.")
        return None

    try:
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is not installed in the active environment. "
            "Install project dependencies and retry."
        ) from exc

    tracking_uri = resolve_mlflow_tracking_uri(training_config)
    mlflow.set_tracking_uri(tracking_uri)

    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        client.search_experiments(max_results=1)
        mlflow.set_experiment(training_config.mlflow.experiment_name)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to reach the MLflow tracking server at {tracking_uri}. "
            "Start the local MLflow server and retry."
        ) from exc

    run_suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{training_config.mlflow.run_name_prefix}_{run_suffix}"

    LOGGER.info("Logging training run to MLflow experiment '%s'", training_config.mlflow.experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "model_name": training_config.model_name,
                "target_column": training_config.features.target_column,
                "test_size": training_config.split.test_size,
                "random_state": training_config.split.random_state,
                "stratify": training_config.split.stratify,
                "training_features": ",".join(training_config.features.training_features),
                "categorical_features": ",".join(training_config.features.categorical_features),
            }
        )
        mlflow.log_params(
            {f"model_{key}": value for key, value in training_config.model_params.items()}
        )
        mlflow.log_metrics(training_result.metrics)
        mlflow.sklearn.log_model(training_result.model, artifact_path="model")
        mlflow.log_dict(config_snapshot, "config_snapshot.yaml")
        mlflow.log_dict(training_result.preprocessing_metadata, "preprocessing_metadata.json")

        if training_config.output.metrics_path is not None and training_config.output.metrics_path.exists():
            mlflow.log_artifact(str(training_config.output.metrics_path))

        model_uri = f"runs:/{run.info.run_id}/model"
        return tracking_uri, model_uri


def register_model_in_mlflow(training_config, tracking_uri: str, model_uri: str) -> None:
    """Register the logged MLflow model into the local MLflow Model Registry."""

    registered_model_name = training_config.mlflow.registered_model_name

    try:
        import mlflow
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is not installed in the active environment. "
            "Install project dependencies and retry."
        ) from exc

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        client.create_registered_model(registered_model_name)
        LOGGER.info("Created registered model '%s'", registered_model_name)
    except MlflowException as exc:
        if "RESOURCE_ALREADY_EXISTS" not in str(exc):
            raise RuntimeError(
                f"Failed to create registered model '{registered_model_name}'."
            ) from exc

    try:
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to register model URI '{model_uri}' under '{registered_model_name}'."
        ) from exc

    LOGGER.info(
        "Registered MLflow model '%s' as version %s",
        registered_model_name,
        model_version.version,
    )


def main() -> None:
    """Run local baseline model training from a YAML config."""

    setup_logging()
    args = parse_args()
    config_path = Path(args.config)
    project_root = Path.cwd()

    LOGGER.info("Loading training config from %s", config_path)
    config_dict = load_yaml_config(config_path)
    log_resolved_config_paths(config_dict, project_root)
    training_config = build_training_config(config_dict, base_dir=project_root)

    trainer = GenericBinaryClassifierTrainer(training_config)
    result = trainer.run()
    mlflow_run_info = log_training_run_to_mlflow(training_config, result, config_dict)
    if training_config.mlflow is not None and mlflow_run_info is not None:
        tracking_uri, model_uri = mlflow_run_info
        if should_skip_mlflow_registration():
            LOGGER.info("Skipping MLflow model registration because SKIP_MLFLOW_REGISTRATION is enabled.")
        else:
            register_model_in_mlflow(training_config, tracking_uri, model_uri)

    LOGGER.info("Training completed. Metrics: %s", result.metrics)
    LOGGER.info("Model bundle saved to %s", result.artifact_path)


if __name__ == "__main__":
    main()

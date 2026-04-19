"""CLI entrypoint for local baseline customer churn training."""

from __future__ import annotations

import argparse
import logging
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

    LOGGER.info("Training completed. Metrics: %s", result.metrics)
    LOGGER.info("Model bundle saved to %s", result.artifact_path)


if __name__ == "__main__":
    main()

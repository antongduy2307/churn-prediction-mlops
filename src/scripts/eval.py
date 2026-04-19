"""CLI entrypoint for standalone local evaluation of saved churn models."""

from __future__ import annotations

import argparse
import logging

from src.model.evaluator import GenericBinaryClassifierEvaluator

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
        description="Evaluate a saved local churn model bundle against a dataset."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the saved model bundle pickle file.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the evaluation dataset (.parquet or .csv).",
    )
    parser.add_argument(
        "--report-dir",
        required=True,
        help="Directory where evaluation reports will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run standalone evaluation for a saved local model bundle."""

    setup_logging()
    args = parse_args()

    evaluator = GenericBinaryClassifierEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        report_dir=args.report_dir,
    )
    result = evaluator.evaluate()

    LOGGER.info("Evaluation metrics: %s", result.metrics)
    LOGGER.info("Saved metrics to %s", result.artifacts.metrics_path)
    LOGGER.info("Saved predictions to %s", result.artifacts.predictions_path)
    LOGGER.info("Saved confusion matrix to %s", result.artifacts.confusion_matrix_path)
    if result.artifacts.feature_importance_path is not None:
        LOGGER.info("Saved feature importance plot to %s", result.artifacts.feature_importance_path)
    else:
        LOGGER.info("Feature importance plot was not generated for the loaded model.")


if __name__ == "__main__":
    main()

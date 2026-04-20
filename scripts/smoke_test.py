"""Local smoke test for core project wiring."""

from __future__ import annotations

import importlib
import argparse
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_BUNDLE_PATH = PROJECT_ROOT / "models" / "random_forest_bundle.pkl"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class CheckResult:
    """Single smoke-test check result."""

    name: str
    status: str
    detail: str
    critical: bool = True


def setup_logging() -> None:
    """Configure logging for CLI execution."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run local or CI-safe smoke checks for the project."
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Enable CI-safe mode and skip local artifact-dependent checks.",
    )
    return parser.parse_args()


def check_import(module_name: str) -> CheckResult:
    """Check that a module can be imported."""

    try:
        importlib.import_module(module_name)
        return CheckResult(module_name, "PASS", "import succeeded")
    except Exception as exc:
        return CheckResult(module_name, "FAIL", f"import failed: {exc}")


def check_path_exists(path: Path) -> CheckResult:
    """Check that a required local path exists."""

    if path.exists():
        return CheckResult(str(path.relative_to(PROJECT_ROOT)), "PASS", "path exists")
    return CheckResult(str(path.relative_to(PROJECT_ROOT)), "FAIL", "path missing")


def skip_check(name: str, detail: str) -> CheckResult:
    """Return a skipped smoke-test check result."""

    return CheckResult(name, "SKIP", detail, critical=False)


def check_bundle_metadata(path: Path) -> CheckResult:
    """Check that the local bundle contains required serving metadata."""

    try:
        with path.open("rb") as file:
            bundle = pickle.load(file)
    except Exception as exc:
        return CheckResult("bundle_metadata", "FAIL", f"bundle load failed: {exc}")

    if not isinstance(bundle, dict):
        return CheckResult("bundle_metadata", "FAIL", "bundle is not a dictionary payload")

    required_keys = {"training_features", "categorical_features", "target_column", "label_encoders"}
    missing_keys = sorted(required_keys - set(bundle.keys()))
    if missing_keys:
        return CheckResult(
            "bundle_metadata",
            "FAIL",
            f"missing bundle keys: {missing_keys}",
        )

    if "model" not in bundle:
        return CheckResult(
            "bundle_metadata",
            "FAIL",
            "bundle does not contain a model field",
        )

    return CheckResult("bundle_metadata", "PASS", "bundle metadata keys are present")


def check_mlflow_serving_config() -> CheckResult:
    """Check that serving MLflow defaults are readable without live calls."""

    try:
        load_model = importlib.import_module("src.serving.load_model")
        model_uri = load_model.DEFAULT_MODEL_URI
        tracking_uri = load_model.DEFAULT_MLFLOW_TRACKING_URI
    except Exception as exc:
        return CheckResult("mlflow_serving_config", "FAIL", f"config import failed: {exc}")

    if not isinstance(model_uri, str) or not model_uri:
        return CheckResult("mlflow_serving_config", "FAIL", "MODEL_URI default is not a non-empty string")
    if not isinstance(tracking_uri, str) or not tracking_uri:
        return CheckResult(
            "mlflow_serving_config",
            "FAIL",
            "MLFLOW_TRACKING_URI default is not a non-empty string",
        )

    return CheckResult(
        "mlflow_serving_config",
        "PASS",
        f"MODEL_URI={model_uri}, MLFLOW_TRACKING_URI={tracking_uri}",
    )


def check_fastapi_app() -> CheckResult:
    """Check that the FastAPI app object can be imported."""

    try:
        main_module = importlib.import_module("api.main")
        app = getattr(main_module, "app", None)
    except Exception as exc:
        return CheckResult("fastapi_app", "FAIL", f"api.main import failed: {exc}")

    if app is None:
        return CheckResult("fastapi_app", "FAIL", "api.main.app is missing")

    return CheckResult("fastapi_app", "PASS", "api.main.app imported successfully")


def build_checks(ci_mode: bool = False) -> list[CheckResult]:
    """Build and execute all smoke-test checks."""

    checks: list[CheckResult] = []

    for module_name in [
        "api.main",
        "src.serving.load_model",
        "src.serving.pre_processing",
        "src.serving.feast_retrieval",
        "src.serving.monitoring",
    ]:
        checks.append(check_import(module_name))

    for path in [
        PROJECT_ROOT / "configs" / "random_forest.yaml",
        PROJECT_ROOT / "feature_repo",
        PROJECT_ROOT / "feature_repo" / "feature_store.yaml",
        PROJECT_ROOT / "feature_repo" / "churn_entities.py",
        PROJECT_ROOT / "feature_repo" / "data_sources.py",
        PROJECT_ROOT / "feature_repo" / "feature_views.py",
    ]:
        checks.append(check_path_exists(path))

    artifact_paths = [
        LOCAL_BUNDLE_PATH,
        PROJECT_ROOT / "data" / "processed" / "processed_churn_data.parquet",
    ]
    for path in artifact_paths:
        if ci_mode:
            checks.append(
                skip_check(
                    str(path.relative_to(PROJECT_ROOT)),
                    "skipped in CI mode because local runtime artifact is not committed",
                )
            )
        else:
            checks.append(check_path_exists(path))

    if ci_mode:
        checks.append(
            skip_check(
                "bundle_metadata",
                "skipped in CI mode because local runtime bundle is not committed",
            )
        )
    else:
        checks.append(check_bundle_metadata(LOCAL_BUNDLE_PATH))

    checks.append(check_mlflow_serving_config())
    checks.append(check_fastapi_app())

    return checks


def print_summary(results: list[CheckResult]) -> None:
    """Print a compact PASS/FAIL summary."""

    for result in results:
        LOGGER.info("[%s] %s: %s", result.status, result.name, result.detail)

    failed = [result for result in results if result.critical and result.status == "FAIL"]
    overall_status = "PASS" if not failed else "FAIL"
    passed_count = sum(result.status == "PASS" for result in results)
    skipped_count = sum(result.status == "SKIP" for result in results)
    LOGGER.info("")
    LOGGER.info("Overall status: %s", overall_status)
    LOGGER.info(
        "Checks run: %s | Passed: %s | Failed: %s | Skipped: %s",
        len(results),
        passed_count,
        len(failed),
        skipped_count,
    )


def main() -> int:
    """Run the local smoke-test suite."""

    setup_logging()
    args = parse_args()
    results = build_checks(ci_mode=args.ci)
    print_summary(results)

    failed = [result for result in results if result.critical and result.status == "FAIL"]
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

"""Local smoke test for core project wiring."""

from __future__ import annotations

import importlib
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
    passed: bool
    detail: str
    critical: bool = True


def setup_logging() -> None:
    """Configure logging for CLI execution."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def check_import(module_name: str) -> CheckResult:
    """Check that a module can be imported."""

    try:
        importlib.import_module(module_name)
        return CheckResult(module_name, True, "import succeeded")
    except Exception as exc:
        return CheckResult(module_name, False, f"import failed: {exc}")


def check_path_exists(path: Path) -> CheckResult:
    """Check that a required local path exists."""

    if path.exists():
        return CheckResult(str(path.relative_to(PROJECT_ROOT)), True, "path exists")
    return CheckResult(str(path.relative_to(PROJECT_ROOT)), False, "path missing")


def check_bundle_metadata(path: Path) -> CheckResult:
    """Check that the local bundle contains required serving metadata."""

    try:
        with path.open("rb") as file:
            bundle = pickle.load(file)
    except Exception as exc:
        return CheckResult("bundle_metadata", False, f"bundle load failed: {exc}")

    if not isinstance(bundle, dict):
        return CheckResult("bundle_metadata", False, "bundle is not a dictionary payload")

    required_keys = {"training_features", "categorical_features", "target_column", "label_encoders"}
    missing_keys = sorted(required_keys - set(bundle.keys()))
    if missing_keys:
        return CheckResult(
            "bundle_metadata",
            False,
            f"missing bundle keys: {missing_keys}",
        )

    if "model" not in bundle:
        return CheckResult(
            "bundle_metadata",
            False,
            "bundle does not contain a model field",
        )

    return CheckResult("bundle_metadata", True, "bundle metadata keys are present")


def check_mlflow_serving_config() -> CheckResult:
    """Check that serving MLflow defaults are readable without live calls."""

    try:
        load_model = importlib.import_module("src.serving.load_model")
        model_uri = load_model.DEFAULT_MODEL_URI
        tracking_uri = load_model.DEFAULT_MLFLOW_TRACKING_URI
    except Exception as exc:
        return CheckResult("mlflow_serving_config", False, f"config import failed: {exc}")

    if not isinstance(model_uri, str) or not model_uri:
        return CheckResult("mlflow_serving_config", False, "MODEL_URI default is not a non-empty string")
    if not isinstance(tracking_uri, str) or not tracking_uri:
        return CheckResult(
            "mlflow_serving_config",
            False,
            "MLFLOW_TRACKING_URI default is not a non-empty string",
        )

    return CheckResult(
        "mlflow_serving_config",
        True,
        f"MODEL_URI={model_uri}, MLFLOW_TRACKING_URI={tracking_uri}",
    )


def check_fastapi_app() -> CheckResult:
    """Check that the FastAPI app object can be imported."""

    try:
        main_module = importlib.import_module("api.main")
        app = getattr(main_module, "app", None)
    except Exception as exc:
        return CheckResult("fastapi_app", False, f"api.main import failed: {exc}")

    if app is None:
        return CheckResult("fastapi_app", False, "api.main.app is missing")

    return CheckResult("fastapi_app", True, "api.main.app imported successfully")


def build_checks() -> list[CheckResult]:
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
        LOCAL_BUNDLE_PATH,
        PROJECT_ROOT / "data" / "processed" / "processed_churn_data.parquet",
        PROJECT_ROOT / "feature_repo" / "feature_store.yaml",
        PROJECT_ROOT / "feature_repo" / "churn_entities.py",
        PROJECT_ROOT / "feature_repo" / "data_sources.py",
        PROJECT_ROOT / "feature_repo" / "feature_views.py",
    ]:
        checks.append(check_path_exists(path))

    checks.append(check_bundle_metadata(LOCAL_BUNDLE_PATH))
    checks.append(check_mlflow_serving_config())
    checks.append(check_fastapi_app())

    return checks


def print_summary(results: list[CheckResult]) -> None:
    """Print a compact PASS/FAIL summary."""

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        LOGGER.info("[%s] %s: %s", status, result.name, result.detail)

    failed = [result for result in results if result.critical and not result.passed]
    overall_status = "PASS" if not failed else "FAIL"
    LOGGER.info("")
    LOGGER.info("Overall status: %s", overall_status)
    LOGGER.info(
        "Checks run: %s | Passed: %s | Failed: %s",
        len(results),
        sum(result.passed for result in results),
        len(failed),
    )


def main() -> int:
    """Run the local smoke-test suite."""

    setup_logging()
    results = build_checks()
    print_summary(results)

    failed = [result for result in results if result.critical and not result.passed]
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

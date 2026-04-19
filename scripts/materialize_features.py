"""Run `feast materialize-incremental` from the local feature repository directory."""

from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for CLI usage."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def current_utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO-8601 format for Feast materialization."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    """Execute `feast materialize-incremental` from inside `feature_repo/`."""

    setup_logging()
    project_root = Path(__file__).resolve().parents[1]
    feature_repo_path = project_root / "feature_repo"
    feast_executable = shutil.which("feast")

    if feast_executable is None:
        raise FileNotFoundError(
            "Feast CLI was not found on PATH. Install dependencies first, then rerun."
        )

    timestamp = current_utc_timestamp()
    command = [feast_executable, "materialize-incremental", timestamp]
    LOGGER.info("Running command from %s: %s", feature_repo_path, " ".join(command))
    subprocess.run(command, cwd=feature_repo_path, check=True)


if __name__ == "__main__":
    main()

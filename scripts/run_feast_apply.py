"""Run `feast apply` from the local feature repository directory."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for CLI usage."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """Execute `feast apply` from inside `feature_repo/`."""

    setup_logging()
    project_root = Path(__file__).resolve().parents[1]
    feature_repo_path = project_root / "feature_repo"
    feast_executable = shutil.which("feast")

    if feast_executable is None:
        raise FileNotFoundError(
            "Feast CLI was not found on PATH. Install dependencies first, then rerun."
        )

    command = [feast_executable, "apply"]
    LOGGER.info("Running command from %s: %s", feature_repo_path, " ".join(command))
    subprocess.run(command, cwd=feature_repo_path, check=True)


if __name__ == "__main__":
    main()

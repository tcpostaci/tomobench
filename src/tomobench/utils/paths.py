"""Path helpers for locating important project directories."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "config" / "benchmark_config.yaml"


def get_repo_root() -> Path:
    """Return the repository root path."""
    return REPO_ROOT


def get_config_path() -> Path:
    """Return the path to the central benchmark configuration file."""
    return CONFIG_PATH

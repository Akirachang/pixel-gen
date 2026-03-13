"""Central configuration for the pixel-gen project."""

import json
from pathlib import Path


# ── Global paths ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ── Global defaults ────────────────────────────────────
SEED = 42
SCRAPE_DELAY = 0.2  # seconds between requests (be polite)


def load_dataset_config(dataset_name: str) -> dict:
    """Load a dataset's config from its dataset.json file.

    Args:
        dataset_name: Name of the dataset folder under data/.

    Returns:
        Dict with dataset-specific settings (sprite_size, palette_size, etc.).
    """
    path = DATA_DIR / dataset_name / "dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"No dataset.json found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_paths(dataset_name: str) -> dict[str, Path]:
    """Return standard paths for a dataset.

    Args:
        dataset_name: Name of the dataset folder under data/.

    Returns:
        Dict with keys: root, raw, processed, manifest.
    """
    root = DATA_DIR / dataset_name
    return {
        "root": root,
        "raw": root / "raw",
        "processed": root / "processed",
        "manifest": root / "manifest.json",
    }

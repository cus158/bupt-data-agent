"""Stable project resource paths for the src-layout package."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
ENV_FILE = PROJECT_ROOT / ".env"
DB_PATH = DATA_DIR / "business.db"

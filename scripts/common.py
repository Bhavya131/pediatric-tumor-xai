"""Shared helpers for the pipeline scripts."""
from pathlib import Path
import yaml


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

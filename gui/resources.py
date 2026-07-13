"""Application resource paths."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def application_icon_path() -> Path:
    suffix = ".ico" if os.name == "nt" else ".png"
    return project_root() / "assets" / f"actr_icon{suffix}"

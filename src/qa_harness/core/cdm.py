"""Загрузка CDM-фикстур вакансий (инфраструктура, доменно-нейтрально).

Общий модуль вместо дублирования load_cdm_files/load_json по раннерам.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_cdm_files(cdm_dir: Path, cdm_count: Optional[int] = None) -> List[Path]:
    """Список cdm_*.json в каталоге (с fallback в подкаталог std/), опц. первые N."""
    paths = [Path(p) for p in sorted(glob.glob(str(Path(cdm_dir) / "cdm_*.json")))]
    if not paths and (Path(cdm_dir) / "std").is_dir():
        paths = [Path(p) for p in sorted(glob.glob(str(Path(cdm_dir) / "std" / "cdm_*.json")))]
    if not paths:
        raise FileNotFoundError(f"No cdm_*.json found in: {cdm_dir}")
    return paths[:cdm_count] if cdm_count else paths

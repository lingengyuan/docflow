from __future__ import annotations

import sysconfig
from pathlib import Path


def resource_candidates() -> list[Path]:
    source_root = Path(__file__).resolve().parents[1]
    candidates = [
        source_root,
        source_root / "share" / "docflow",
    ]
    data_path = sysconfig.get_paths().get("data")
    if data_path:
        candidates.append(Path(data_path) / "share" / "docflow")
    return candidates


def resource_path(*parts: str) -> Path:
    for base in resource_candidates():
        candidate = base.joinpath(*parts)
        if candidate.exists():
            return candidate
    return resource_candidates()[0].joinpath(*parts)

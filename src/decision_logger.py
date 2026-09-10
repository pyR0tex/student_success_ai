from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(records: list[dict], path: Path) -> Path:
    """Write one structured decision record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path

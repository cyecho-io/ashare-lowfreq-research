from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class DatasetSummary:
    name: str
    path: str
    rows: int
    min_date: str | None = None
    max_date: str | None = None


@dataclass(frozen=True)
class StorageCatalog:
    schema_version: str
    source_type: str
    source_path: str
    imported_at: str
    datasets: list[DatasetSummary]
    sqlite_summary: dict[str, int | str] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def write_catalog(path: str | Path, catalog: StorageCatalog) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(catalog.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def build_catalog(
    source_type: str,
    source_path: str,
    datasets: list[DatasetSummary],
    sqlite_summary: dict[str, int | str] | None = None,
    schema_version: str = "v1",
) -> StorageCatalog:
    return StorageCatalog(
        schema_version=schema_version,
        source_type=source_type,
        source_path=source_path,
        imported_at=datetime.now().isoformat(timespec="seconds"),
        datasets=datasets,
        sqlite_summary=sqlite_summary,
    )

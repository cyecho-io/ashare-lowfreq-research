from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from ashare_backtest.sandbox import StrategyValidationError, StrategyValidator


@dataclass(frozen=True)
class StrategyRegistration:
    strategy_id: str
    name: str
    class_name: str
    file_name: str
    sha256: str


class StrategyLibrary:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self.validator = StrategyValidator()

    def register(self, source_path: str | Path) -> StrategyRegistration:
        report = self.validator.validate_file(source_path)
        source = Path(source_path)
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        strategy_id = digest[:12]
        target_name = f"{strategy_id}_{source.name}"
        target_path = self.root / target_name
        shutil.copy2(source, target_path)

        record = StrategyRegistration(
            strategy_id=strategy_id,
            name=source.stem,
            class_name=report.class_name,
            file_name=target_name,
            sha256=digest,
        )
        self._save_record(record)
        return record

    def list(self) -> list[StrategyRegistration]:
        if not self.index_path.exists():
            return []
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        return [StrategyRegistration(**item) for item in data]

    def _save_record(self, record: StrategyRegistration) -> None:
        records = self.list()
        if any(item.sha256 == record.sha256 for item in records):
            raise StrategyValidationError("strategy script already registered")
        records.append(record)
        payload = [asdict(item) for item in records]
        self.index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

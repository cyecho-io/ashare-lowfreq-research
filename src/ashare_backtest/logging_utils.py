from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "results" / "logs"
DEFAULT_LOG_PATH = DEFAULT_LOG_DIR / "ashare_web.log"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5

_CONFIGURED = False


def configure_file_logging(
    *,
    log_path: str | Path | None = None,
    level: int = logging.INFO,
) -> Path:
    global _CONFIGURED

    target = Path(log_path) if log_path is not None else DEFAULT_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("ashare_backtest")
    if _CONFIGURED:
        return target

    max_bytes = int(os.environ.get("ASHARE_LOG_MAX_BYTES", str(DEFAULT_MAX_BYTES)))
    backup_count = int(os.environ.get("ASHARE_LOG_BACKUP_COUNT", str(DEFAULT_BACKUP_COUNT)))

    handler = RotatingFileHandler(
        target,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    root_logger.propagate = False
    _CONFIGURED = True
    root_logger.info(
        "file logging configured path=%s max_bytes=%s backup_count=%s",
        target.as_posix(),
        max_bytes,
        backup_count,
    )
    return target


def get_logger(name: str) -> logging.Logger:
    configure_file_logging()
    return logging.getLogger(f"ashare_backtest.{name}")

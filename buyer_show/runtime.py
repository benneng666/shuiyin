"""Desktop settings, local logs, and a process-scoped single-instance lock."""
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path


def app_data_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share") / "BuyerShowValidator"


def _validated_settings(data):
    defaults = {"output": str(Path.home() / "Downloads" / "买家秀"), "max_scrolls": 200, "channel": "msedge"}
    if not isinstance(data, dict):
        return defaults
    output = data.get("output")
    if isinstance(output, str) and output.strip() and "\x00" not in output:
        defaults["output"] = output
    limit = data.get("max_scrolls")
    if type(limit) is int and 1 <= limit <= 1000:
        defaults["max_scrolls"] = limit
    if data.get("channel") in {"msedge", "chrome"}:
        defaults["channel"] = data["channel"]
    return defaults


def load_settings() -> dict:
    try:
        data = json.loads((app_data_dir() / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return _validated_settings(data)


def save_settings(data: dict):
    base = app_data_dir()
    base.mkdir(parents=True, exist_ok=True)
    temp = base / "settings.json.tmp"
    temp.write_text(json.dumps(_validated_settings(data), ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(base / "settings.json")


def setup_logging():
    logdir = app_data_dir() / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("buyer_show")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = RotatingFileHandler(logdir / "app.log", maxBytes=512_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


class InstanceLock:
    """OS releases the lock on exit, even after a crash. No stale PID checks."""

    def __init__(self, path: Path | None = None):
        self.path = path or app_data_dir() / "desktop.lock"
        self.file = None

    def acquire(self) -> bool:
        if self.file:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file = self.path.open("a+b")
        try:
            file.seek(0, os.SEEK_END)
            if not file.tell():
                file.write(b"0")
                file.flush()
            file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            file.close()
            return False
        self.file = file
        return True

    def release(self):
        if not self.file:
            return
        try:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        finally:
            self.file.close()
            self.file = None

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SpecValidationLogger:
    """Thread-safe JSONL logger for speculative validation diagnostics."""

    def __init__(self, enabled: bool = True, directory: Optional[str] = None):
        self.enabled = enabled
        self.directory = directory or os.getcwd()
        self._lock = threading.Lock()
        self._file = None
        self._path = None
        if self.enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_log.{timestamp}.jsonl"
            self._path = os.path.join(self.directory, filename)

    @property
    def path(self) -> Optional[str]:
        return self._path

    def log(self, entry: Dict[str, Any]) -> None:
        if not self.enabled or self._path is None:
            return

        line = json.dumps(entry, ensure_ascii=False, default=str)

        try:
            with self._lock:
                if self._file is None:
                    os.makedirs(os.path.dirname(self._path), exist_ok=True)
                    self._file = open(self._path, "a", encoding="utf-8")
                self._file.write(line + "\n")
                self._file.flush()
        except OSError as exc:
            logger.warning("Failed to write validation log entry: %s", exc)

    def close(self) -> None:
        if self._file is None:
            return
        with self._lock:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()

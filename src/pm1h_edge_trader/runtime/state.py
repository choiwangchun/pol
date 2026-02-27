from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger("pm1h_edge_trader.runtime_state")


class RuntimeStateManager:
    """Manages runtime/policy checkpoint files with atomic writes."""

    def __init__(
        self,
        *,
        state_dir: Path,
        runtime_file_name: str = "runtime_state.json",
        policy_file_name: str = "policy_state.json",
    ) -> None:
        self._state_dir = Path(state_dir)
        self._runtime_path = self._state_dir / runtime_file_name
        self._policy_path = self._state_dir / policy_file_name

    @property
    def runtime_path(self) -> Path:
        return self._runtime_path

    @property
    def policy_path(self) -> Path:
        return self._policy_path

    def load_runtime(self) -> dict[str, Any]:
        return _load_json_object(self._runtime_path)

    def save_runtime(self, payload: Mapping[str, Any]) -> None:
        _atomic_write_json(self._runtime_path, payload)

    def load_policy(self) -> dict[str, Any]:
        return _load_json_object(self._policy_path)

    def save_policy(self, payload: Mapping[str, Any]) -> None:
        _atomic_write_json(self._policy_path, payload)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("state_load_failed path=%s error=%s", path, exc)
        return {}
    if isinstance(payload, dict):
        return payload
    LOGGER.warning(
        "state_load_invalid_payload path=%s payload_type=%s",
        path,
        type(payload).__name__,
    )
    return {}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    data = json.dumps(dict(payload), ensure_ascii=True, indent=2)

    with tmp_path.open("w", encoding="utf-8") as fp:
        fp.write(data)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp_path, path)

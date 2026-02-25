from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PositionSnapshot:
    up_size: float
    down_size: float
    up_value: float
    down_value: float
    has_any_position_above_threshold: bool


def extract_position_snapshot(
    positions: Sequence[Mapping[str, Any]],
    *,
    up_token_id: str,
    down_token_id: str,
    size_threshold: float,
) -> PositionSnapshot:
    up_key = str(up_token_id).strip()
    down_key = str(down_token_id).strip()
    threshold = max(0.0, float(size_threshold))

    up_size = 0.0
    down_size = 0.0
    up_value = 0.0
    down_value = 0.0
    for row in positions:
        token_id = _extract_token_id(row)
        if token_id is None:
            continue
        size = _safe_float(row.get("size")) or 0.0
        value = _safe_float(row.get("currentValue")) or 0.0
        if token_id == up_key:
            up_size += max(0.0, size)
            up_value += max(0.0, value)
        elif token_id == down_key:
            down_size += max(0.0, size)
            down_value += max(0.0, value)

    has_any = (up_size > threshold) or (down_size > threshold)
    return PositionSnapshot(
        up_size=up_size,
        down_size=down_size,
        up_value=up_value,
        down_value=down_value,
        has_any_position_above_threshold=has_any,
    )


def evaluate_position_mismatch(
    *,
    snapshot: PositionSnapshot,
    local_has_open_exposure: bool,
) -> bool:
    if not snapshot.has_any_position_above_threshold:
        return False
    return not local_has_open_exposure


def _extract_token_id(payload: Mapping[str, Any]) -> str | None:
    for key in ("asset", "asset_id", "token_id", "tokenId", "clobTokenId", "id"):
        value = payload.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
        if isinstance(value, Mapping):
            nested = _extract_token_id(value)
            if nested is not None:
                return nested
    return None


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None

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


@dataclass(frozen=True)
class PositionMismatchResult:
    mismatch: bool
    reason: str
    wallet_up_size: float
    wallet_down_size: float
    local_up_size: float
    local_down_size: float
    up_diff: float
    down_diff: float


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
    local_up_size: float,
    local_down_size: float,
    size_threshold: float,
    relative_tolerance: float = 0.05,
) -> PositionMismatchResult:
    threshold = max(0.0, float(size_threshold))
    rel_tol = max(0.0, float(relative_tolerance))

    wallet_up = max(0.0, float(snapshot.up_size))
    wallet_down = max(0.0, float(snapshot.down_size))
    local_up = max(0.0, float(local_up_size))
    local_down = max(0.0, float(local_down_size))

    wallet_has = (wallet_up > threshold) or (wallet_down > threshold)
    local_has = (local_up > threshold) or (local_down > threshold)

    up_diff = abs(wallet_up - local_up)
    down_diff = abs(wallet_down - local_down)

    if wallet_has and not local_has:
        return PositionMismatchResult(
            mismatch=True,
            reason="wallet_only_exposure",
            wallet_up_size=wallet_up,
            wallet_down_size=wallet_down,
            local_up_size=local_up,
            local_down_size=local_down,
            up_diff=up_diff,
            down_diff=down_diff,
        )
    if local_has and not wallet_has:
        return PositionMismatchResult(
            mismatch=True,
            reason="local_only_exposure",
            wallet_up_size=wallet_up,
            wallet_down_size=wallet_down,
            local_up_size=local_up,
            local_down_size=local_down,
            up_diff=up_diff,
            down_diff=down_diff,
        )

    if not wallet_has and not local_has:
        return PositionMismatchResult(
            mismatch=False,
            reason="none",
            wallet_up_size=wallet_up,
            wallet_down_size=wallet_down,
            local_up_size=local_up,
            local_down_size=local_down,
            up_diff=up_diff,
            down_diff=down_diff,
        )

    up_tol = max(threshold, rel_tol * max(wallet_up, local_up))
    down_tol = max(threshold, rel_tol * max(wallet_down, local_down))
    mismatch = up_diff > up_tol or down_diff > down_tol
    return PositionMismatchResult(
        mismatch=mismatch,
        reason="size_divergence" if mismatch else "within_tolerance",
        wallet_up_size=wallet_up,
        wallet_down_size=wallet_down,
        local_up_size=local_up,
        local_down_size=local_down,
        up_diff=up_diff,
        down_diff=down_diff,
    )


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

"""Live auto-claim utilities for redeemable Polymarket positions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol, TYPE_CHECKING
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from eth_account import Account

if TYPE_CHECKING:
    from .config import PolymarketLiveAuthConfig
    from .execution.polymarket_live_adapter import PolymarketLiveExecutionAdapter


CTF_REDEEM_POSITIONS_ABI: list[dict[str, Any]] = [
    {
        "inputs": [
            {"internalType": "address", "name": "collateralToken", "type": "address"},
            {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256[]", "name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]


@dataclass(slots=True, frozen=True)
class RedeemTarget:
    condition_id: str
    index_sets: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class ClaimAttempt:
    condition_id: str
    index_sets: tuple[int, ...]
    success: bool
    tx_hash: str | None
    error: str | None


@dataclass(slots=True, frozen=True)
class ClaimRunSummary:
    checked_positions: int
    targets: int
    attempted: int
    claimed: int
    errors: int
    skipped_cooldown: int
    attempts: tuple[ClaimAttempt, ...]


class RedeemablePositionSource(Protocol):
    def fetch_redeemable_positions(self, *, user_address: str, size_threshold: float) -> list[Mapping[str, Any]]:
        """Returns redeemable position payloads for one user."""


class RedeemExecutor(Protocol):
    def redeem_positions(self, target: RedeemTarget) -> ClaimAttempt:
        """Executes redeemPositions for one condition target."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def extract_redeemable_targets(
    positions: Sequence[Mapping[str, Any]],
    *,
    size_threshold: float,
) -> list[RedeemTarget]:
    grouped: dict[str, set[int]] = {}
    threshold = max(0.0, float(size_threshold))
    for position in positions:
        if not _as_bool(position.get("redeemable")):
            continue
        size = _safe_float(position.get("size"))
        if size is None or size <= threshold:
            continue
        condition_id = _normalize_condition_id(position.get("conditionId") or position.get("condition_id"))
        if condition_id is None:
            continue
        index_set = _extract_index_set(position)
        if index_set is None or index_set <= 0:
            continue
        grouped.setdefault(condition_id, set()).add(index_set)
    targets = [
        RedeemTarget(condition_id=condition_id, index_sets=tuple(sorted(index_sets)))
        for condition_id, index_sets in grouped.items()
    ]
    targets.sort(key=lambda item: item.condition_id)
    return targets


class DataApiRedeemablePositionSource:
    def __init__(self, *, base_url: str = "https://data-api.polymarket.com", timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = max(1.0, float(timeout_seconds))

    def fetch_redeemable_positions(self, *, user_address: str, size_threshold: float) -> list[Mapping[str, Any]]:
        normalized_user = _normalize_address(user_address)
        if normalized_user is None:
            raise RuntimeError("Invalid user address for redeemable position lookup.")
        limit = 500
        offset = 0
        rows: list[Mapping[str, Any]] = []
        while True:
            query = urlencode(
                {
                    "user": normalized_user,
                    "redeemable": "true",
                    "sizeThreshold": f"{max(0.0, float(size_threshold)):.8f}",
                    "limit": str(limit),
                    "offset": str(offset),
                }
            )
            url = f"{self._base_url}/positions?{query}"
            payload = _http_get_json(url, timeout_seconds=self._timeout_seconds)
            if not isinstance(payload, list):
                break
            page: list[Mapping[str, Any]] = [item for item in payload if isinstance(item, Mapping)]
            if not page:
                break
            rows.extend(page)
            if len(page) < limit:
                break
            offset += limit
        return rows


class OnChainRedeemExecutor:
    def __init__(
        self,
        *,
        rpc_url: str,
        chain_id: int,
        private_key: str,
        from_address: str,
        conditional_token_address: str,
        collateral_token_address: str,
        tx_timeout_seconds: float = 120.0,
    ) -> None:
        try:
            from web3 import Web3
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("web3 dependency is required for live auto-claim.") from exc

        self._web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 20}))
        if not self._web3.is_connected():
            raise RuntimeError(f"Failed to connect to polygon RPC: {rpc_url}")

        if not _has_text(private_key):
            raise RuntimeError("Missing private key for auto-claim.")
        account = Account.from_key(private_key)
        normalized_from = _normalize_address(from_address)
        if normalized_from is None:
            raise RuntimeError("Invalid from/funder address for auto-claim.")
        if account.address.lower() != normalized_from.lower():
            raise RuntimeError("Auto-claim requires private key control of funder address.")

        conditional_token = _normalize_address(conditional_token_address)
        collateral_token = _normalize_address(collateral_token_address)
        if conditional_token is None:
            raise RuntimeError("Conditional token contract address is unavailable.")
        if collateral_token is None:
            raise RuntimeError("Collateral token contract address is unavailable.")

        self._account = account
        self._from_address = self._web3.to_checksum_address(normalized_from)
        self._collateral_token = self._web3.to_checksum_address(collateral_token)
        self._chain_id = int(chain_id)
        self._tx_timeout_seconds = max(10.0, float(tx_timeout_seconds))
        self._contract = self._web3.eth.contract(
            address=self._web3.to_checksum_address(conditional_token),
            abi=CTF_REDEEM_POSITIONS_ABI,
        )

    def redeem_positions(self, target: RedeemTarget) -> ClaimAttempt:
        tx_hash_hex: str | None = None
        try:
            condition_id = _normalize_condition_id(target.condition_id)
            if condition_id is None:
                raise RuntimeError("Invalid condition id for redeem transaction.")
            condition_bytes = bytes.fromhex(condition_id[2:])
            fn = self._contract.functions.redeemPositions(
                self._collateral_token,
                bytes(32),
                condition_bytes,
                list(target.index_sets),
            )
            nonce = self._web3.eth.get_transaction_count(self._from_address, "pending")
            tx = fn.build_transaction(
                {
                    "from": self._from_address,
                    "chainId": self._chain_id,
                    "nonce": nonce,
                }
            )
            estimated_gas = self._web3.eth.estimate_gas(tx)
            tx["gas"] = max(150_000, int(estimated_gas * 1.25))
            if "maxFeePerGas" not in tx and "gasPrice" not in tx:
                gas_price = self._web3.eth.gas_price
                if gas_price is not None and int(gas_price) > 0:
                    tx["gasPrice"] = int(gas_price)

            signed = self._account.sign_transaction(tx)
            raw_tx = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
            tx_hash = self._web3.eth.send_raw_transaction(raw_tx)
            tx_hash_hex = tx_hash.hex()
            receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash, timeout=self._tx_timeout_seconds)
            status = int(receipt.get("status", 0)) if isinstance(receipt, Mapping) else int(getattr(receipt, "status", 0))
            if status == 1:
                return ClaimAttempt(
                    condition_id=condition_id,
                    index_sets=target.index_sets,
                    success=True,
                    tx_hash=tx_hash_hex,
                    error=None,
                )
            return ClaimAttempt(
                condition_id=condition_id,
                index_sets=target.index_sets,
                success=False,
                tx_hash=tx_hash_hex,
                error="Transaction receipt status=0",
            )
        except Exception as exc:
            return ClaimAttempt(
                condition_id=target.condition_id,
                index_sets=target.index_sets,
                success=False,
                tx_hash=tx_hash_hex,
                error=f"{exc.__class__.__name__}: {exc}",
            )


class LiveAutoClaimWorker:
    def __init__(
        self,
        *,
        user_address: str,
        position_source: RedeemablePositionSource,
        redeem_executor: RedeemExecutor,
        size_threshold: float = 0.0001,
        cooldown_seconds: float = 600.0,
        now_fn: Callable[[], datetime] = utc_now,
    ) -> None:
        normalized_user = _normalize_address(user_address)
        if normalized_user is None:
            raise RuntimeError("Invalid user address for auto-claim.")
        self._user_address = normalized_user
        self._position_source = position_source
        self._redeem_executor = redeem_executor
        self._size_threshold = max(0.0, float(size_threshold))
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._now_fn = now_fn
        self._cooldowns: dict[str, datetime] = {}

    def run_once(self) -> ClaimRunSummary:
        now = self._now_fn()
        positions = self._position_source.fetch_redeemable_positions(
            user_address=self._user_address,
            size_threshold=self._size_threshold,
        )
        checked_positions = len(positions)
        targets = extract_redeemable_targets(
            positions,
            size_threshold=self._size_threshold,
        )
        attempts: list[ClaimAttempt] = []
        attempted = 0
        claimed = 0
        errors = 0
        skipped_cooldown = 0

        for target in targets:
            key = target.condition_id.lower()
            next_allowed = self._cooldowns.get(key)
            if next_allowed is not None and now < next_allowed:
                skipped_cooldown += 1
                continue

            attempt = self._redeem_executor.redeem_positions(target)
            attempts.append(attempt)
            attempted += 1
            if attempt.success:
                claimed += 1
            else:
                errors += 1
            self._cooldowns[key] = now + timedelta(seconds=self._cooldown_seconds)

        return ClaimRunSummary(
            checked_positions=checked_positions,
            targets=len(targets),
            attempted=attempted,
            claimed=claimed,
            errors=errors,
            skipped_cooldown=skipped_cooldown,
            attempts=tuple(attempts),
        )


def build_live_auto_claim_worker(
    *,
    auth: "PolymarketLiveAuthConfig",
    adapter: "PolymarketLiveExecutionAdapter",
    rpc_url: str,
    data_api_base_url: str,
    size_threshold: float,
    cooldown_seconds: float,
    tx_timeout_seconds: float,
) -> LiveAutoClaimWorker:
    normalized_funder = _normalize_address(auth.funder)
    signer_address = _address_from_private_key(auth.private_key)
    if normalized_funder is None:
        raise RuntimeError("Auto-claim requires POLYMARKET_FUNDER.")
    if signer_address is None:
        raise RuntimeError("Auto-claim requires a valid POLYMARKET_PRIVATE_KEY.")
    if signer_address.lower() != normalized_funder.lower():
        raise RuntimeError("Auto-claim currently supports direct-wallet mode only (funder must equal signer).")

    conditional_token_address = adapter.get_conditional_address()
    collateral_token_address = adapter.get_collateral_address()
    if conditional_token_address is None:
        raise RuntimeError("Failed to fetch conditional token contract address for auto-claim.")
    if collateral_token_address is None:
        raise RuntimeError("Failed to fetch collateral token contract address for auto-claim.")

    source = DataApiRedeemablePositionSource(base_url=data_api_base_url)
    executor = OnChainRedeemExecutor(
        rpc_url=rpc_url,
        chain_id=auth.chain_id,
        private_key=auth.private_key or "",
        from_address=normalized_funder,
        conditional_token_address=conditional_token_address,
        collateral_token_address=collateral_token_address,
        tx_timeout_seconds=tx_timeout_seconds,
    )
    return LiveAutoClaimWorker(
        user_address=normalized_funder,
        position_source=source,
        redeem_executor=executor,
        size_threshold=size_threshold,
        cooldown_seconds=cooldown_seconds,
    )


def _http_get_json(url: str, *, timeout_seconds: float) -> object:
    request = Request(url=url, method="GET", headers={"User-Agent": "pm1h-edge-trader"})
    with urlopen(request, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _extract_index_set(payload: Mapping[str, Any]) -> int | None:
    if "outcomeIndex" in payload:
        outcome_index_raw = payload.get("outcomeIndex")
    else:
        outcome_index_raw = payload.get("outcome_index")
    outcome_index = _safe_int(outcome_index_raw)
    if outcome_index is not None and outcome_index >= 0:
        return 1 << outcome_index
    if "indexSet" in payload:
        index_set_raw = payload.get("indexSet")
    else:
        index_set_raw = payload.get("index_set")
    raw_index_set = _safe_int(index_set_raw)
    if raw_index_set is not None and raw_index_set > 0:
        return raw_index_set
    return None


def _address_from_private_key(private_key: str | None) -> str | None:
    if not _has_text(private_key):
        return None
    try:
        return Account.from_key(private_key).address
    except Exception:
        return None


def _normalize_condition_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if len(normalized) != 66 or not normalized.startswith("0x"):
        return None
    try:
        int(normalized[2:], 16)
    except ValueError:
        return None
    return normalized


def _normalize_address(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if len(normalized) != 42 or not normalized.startswith("0x"):
        return None
    try:
        int(normalized[2:], 16)
    except ValueError:
        return None
    return normalized


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _has_text(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from math import log, sqrt
from pathlib import Path
from random import Random
from typing import Mapping, Protocol

from ..engine import DecisionConfig
from ..logger.models import ExecutionLogRecord


@dataclass(frozen=True)
class PolicyBanditConfig:
    enabled: bool = False
    shadow_mode: bool = False
    exploration_epsilon: float = 0.05
    ucb_c: float = 1.0
    reward_turnover_lambda: float = 0.0
    reward_risk_penalty: float = 0.0
    vol_ratio_threshold: float = 1.1
    spread_tight_threshold: float = 0.03
    dataset_path: Path = Path("logs/policy_dataset.csv")


@dataclass(frozen=True)
class PolicyProfile:
    profile_id: str
    edge_min_delta: float = 0.0
    edge_buffer_delta: float = 0.0
    cost_rate_delta: float = 0.0
    kelly_fraction_multiplier: float = 1.0
    force_no_trade: bool = False


@dataclass(frozen=True)
class PolicySelection:
    timestamp: datetime
    market_id: str
    context_id: str
    chosen_profile_id: str
    applied_profile_id: str
    exploration: bool
    bankroll_at_entry: float
    seconds_to_expiry: float
    rv_long: float
    rv_short: float
    iv: float | None
    sigma: float
    spot: float
    strike: float
    bid_up: float
    ask_up: float
    bid_down: float
    ask_down: float
    spread: float


@dataclass
class _BanditArmStats:
    count: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total_reward / float(self.count)


@dataclass
class _PendingEpisode:
    order_id: str
    selection: PolicySelection
    side: str
    fill_price: float
    fill_size: float
    order_notional: float


class _RandomSource(Protocol):
    def random(self) -> float:
        ...

    def choice(self, values):  # type: ignore[no-untyped-def]
        ...


class PolicyDatasetReporter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists() or self._path.stat().st_size == 0:
            self._write_header()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, row: dict[str, object]) -> None:
        with self._path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.csv_fields())
            writer.writerow(_serialize_row(row))

    def reset(self) -> None:
        self._write_header()

    @staticmethod
    def csv_fields() -> list[str]:
        return [
            "timestamp",
            "market_id",
            "order_id",
            "side",
            "context_id",
            "profile_id",
            "chosen_profile_id",
            "applied_profile_id",
            "exploration",
            "bankroll_at_entry",
            "seconds_to_expiry",
            "rv_long",
            "rv_short",
            "iv",
            "sigma",
            "spot",
            "strike",
            "bid_up",
            "ask_up",
            "bid_down",
            "ask_down",
            "spread",
            "fill_price",
            "fill_size",
            "order_notional",
            "settlement_outcome",
            "pnl",
            "reward",
            "risk_limit_breach",
        ]

    def _write_header(self) -> None:
        with self._path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.csv_fields())
            writer.writeheader()


class PolicyBanditController:
    def __init__(
        self,
        *,
        config: PolicyBanditConfig,
        base_decision_config: DecisionConfig,
        random_source: _RandomSource | None = None,
    ) -> None:
        self._config = config
        self._base = base_decision_config
        self._profiles = _default_profiles()
        self._baseline_profile_id = "P2_BALANCED"
        self._random = random_source or Random()
        self._stats: dict[str, dict[str, _BanditArmStats]] = {}
        self._pending: dict[str, _PendingEpisode] = {}
        self._dataset = PolicyDatasetReporter(config.dataset_path)

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def shadow_mode(self) -> bool:
        return self._config.shadow_mode

    def reset(self) -> None:
        self._stats = {}
        self._pending = {}
        self._dataset.reset()

    def export_state(self) -> dict[str, object]:
        contexts: dict[str, dict[str, dict[str, float | int]]] = {}
        for context_id, profile_map in self._stats.items():
            serialized: dict[str, dict[str, float | int]] = {}
            for profile_id, stats in profile_map.items():
                serialized[profile_id] = {
                    "count": stats.count,
                    "total_reward": stats.total_reward,
                    "mean_reward": stats.mean_reward,
                }
            contexts[context_id] = serialized
        pending: list[dict[str, object]] = []
        for order_id in sorted(self._pending.keys()):
            episode = self._pending[order_id]
            pending.append(
                {
                    "order_id": episode.order_id,
                    "selection": _serialize_selection(episode.selection),
                    "side": episode.side,
                    "fill_price": episode.fill_price,
                    "fill_size": episode.fill_size,
                    "order_notional": episode.order_notional,
                }
            )
        return {
            "updated_at": datetime.utcnow().isoformat(),
            "contexts": contexts,
            "pending": pending,
        }

    def import_state(self, payload: dict[str, object]) -> None:
        self._stats = {}
        contexts = payload.get("contexts")
        if isinstance(contexts, dict):
            for context_id, profile_map in contexts.items():
                if not isinstance(profile_map, dict):
                    continue
                typed_map: dict[str, _BanditArmStats] = {}
                for profile_id, raw in profile_map.items():
                    if not isinstance(raw, dict):
                        continue
                    count = raw.get("count")
                    total_reward = raw.get("total_reward")
                    if not isinstance(count, int):
                        continue
                    if not isinstance(total_reward, (float, int)):
                        continue
                    typed_map[profile_id] = _BanditArmStats(count=count, total_reward=float(total_reward))
                if typed_map:
                    self._stats[str(context_id)] = typed_map

        self._pending = {}
        pending_rows = payload.get("pending")
        if isinstance(pending_rows, list):
            for row in pending_rows:
                if not isinstance(row, Mapping):
                    continue
                order_id = str(row.get("order_id") or "").strip()
                selection_raw = row.get("selection")
                selection = _deserialize_selection(selection_raw)
                side = str(row.get("side") or "").strip()
                fill_price = _safe_float(row.get("fill_price"))
                fill_size = _safe_float(row.get("fill_size"))
                order_notional = _safe_float(row.get("order_notional"))
                if (
                    not order_id
                    or selection is None
                    or not side
                    or fill_price is None
                    or fill_size is None
                    or order_notional is None
                ):
                    continue
                self._pending[order_id] = _PendingEpisode(
                    order_id=order_id,
                    selection=selection,
                    side=side,
                    fill_price=fill_price,
                    fill_size=fill_size,
                    order_notional=order_notional,
                )

    def select_profile(
        self,
        *,
        timestamp: datetime,
        market_id: str,
        seconds_to_expiry: float,
        rv_long: float,
        rv_short: float,
        sigma: float,
        iv: float | None,
        spot: float,
        strike: float,
        bid_up: float,
        ask_up: float,
        bid_down: float,
        ask_down: float,
        bankroll_at_entry: float,
        allow_exploration: bool,
    ) -> PolicySelection:
        spread = max(max(ask_up - bid_up, 0.0), max(ask_down - bid_down, 0.0))
        context_id = build_context_id(
            seconds_to_expiry=seconds_to_expiry,
            rv_long=rv_long,
            rv_short=rv_short,
            max_spread=spread,
            vol_ratio_threshold=self._config.vol_ratio_threshold,
            spread_tight_threshold=self._config.spread_tight_threshold,
        )
        chosen = self._baseline_profile_id
        exploration = False
        if self._config.enabled:
            chosen, exploration = self._choose_profile(
                context_id=context_id,
                allow_exploration=allow_exploration,
            )
        applied = self._baseline_profile_id if self._config.shadow_mode else chosen
        return PolicySelection(
            timestamp=timestamp,
            market_id=market_id,
            context_id=context_id,
            chosen_profile_id=chosen,
            applied_profile_id=applied,
            exploration=exploration,
            bankroll_at_entry=max(0.0, bankroll_at_entry),
            seconds_to_expiry=max(0.0, seconds_to_expiry),
            rv_long=max(0.0, rv_long),
            rv_short=max(0.0, rv_short),
            iv=iv,
            sigma=max(0.0, sigma),
            spot=max(0.0, spot),
            strike=max(0.0, strike),
            bid_up=max(0.0, bid_up),
            ask_up=max(0.0, ask_up),
            bid_down=max(0.0, bid_down),
            ask_down=max(0.0, ask_down),
            spread=spread,
        )

    def decision_config_for(self, profile_id: str) -> DecisionConfig:
        profile = self._profiles.get(profile_id, self._profiles[self._baseline_profile_id])
        if profile.force_no_trade:
            return DecisionConfig(
                sigma_weight=self._base.sigma_weight,
                edge_min=1.0,
                edge_buffer_up=max(self._base.edge_buffer_up, 0.0),
                edge_buffer_down=max(self._base.edge_buffer_down, 0.0),
                cost_rate_up=max(0.0, self._base.cost_rate_up),
                cost_rate_down=max(0.0, self._base.cost_rate_down),
                kelly_fraction=0.0,
                f_cap=self._base.f_cap,
                min_order_size=self._base.min_order_size,
                epsilon=self._base.epsilon,
            )

        edge_min = max(0.0, self._base.edge_min + profile.edge_min_delta)
        edge_buffer_up = max(0.0, self._base.edge_buffer_up + profile.edge_buffer_delta)
        edge_buffer_down = max(0.0, self._base.edge_buffer_down + profile.edge_buffer_delta)
        cost_rate_up = max(0.0, self._base.cost_rate_up + profile.cost_rate_delta)
        cost_rate_down = max(0.0, self._base.cost_rate_down + profile.cost_rate_delta)
        kelly = self._base.kelly_fraction * profile.kelly_fraction_multiplier
        kelly = min(1.0, max(0.0, kelly))
        return DecisionConfig(
            sigma_weight=self._base.sigma_weight,
            edge_min=edge_min,
            edge_buffer_up=edge_buffer_up,
            edge_buffer_down=edge_buffer_down,
            cost_rate_up=cost_rate_up,
            cost_rate_down=cost_rate_down,
            kelly_fraction=kelly,
            f_cap=self._base.f_cap,
            min_order_size=self._base.min_order_size,
            epsilon=self._base.epsilon,
        )

    def on_fill(self, record: ExecutionLogRecord, *, selection: PolicySelection | None) -> None:
        if selection is None:
            return
        order_id = (record.order_id or "").strip()
        if not order_id or order_id == "-":
            return
        order_notional = max(0.0, float(record.price) * float(record.size))
        self._pending[order_id] = _PendingEpisode(
            order_id=order_id,
            selection=selection,
            side=str(record.side or "").strip(),
            fill_price=float(record.price),
            fill_size=float(record.size),
            order_notional=order_notional,
        )

    def on_settlement(self, record: ExecutionLogRecord, *, risk_limit_breach: bool = False) -> float | None:
        order_id = (record.order_id or "").strip()
        episode = self._pending.pop(order_id, None)
        if episode is None:
            return None

        pnl = float(record.pnl) if record.pnl is not None else 0.0
        bankroll = max(episode.selection.bankroll_at_entry, 1e-9)
        turnover = episode.order_notional / bankroll
        reward = (pnl / bankroll) - (self._config.reward_turnover_lambda * turnover)
        if risk_limit_breach:
            reward -= self._config.reward_risk_penalty

        if self._config.enabled:
            profile_id = episode.selection.applied_profile_id
            self._update_stats(
                context_id=episode.selection.context_id,
                profile_id=profile_id,
                reward=reward,
            )
        self._dataset.append(
            {
                "timestamp": record.timestamp.isoformat(),
                "market_id": record.market_id,
                "order_id": order_id,
                "side": episode.side,
                "context_id": episode.selection.context_id,
                "profile_id": episode.selection.applied_profile_id,
                "chosen_profile_id": episode.selection.chosen_profile_id,
                "applied_profile_id": episode.selection.applied_profile_id,
                "exploration": episode.selection.exploration,
                "bankroll_at_entry": episode.selection.bankroll_at_entry,
                "seconds_to_expiry": episode.selection.seconds_to_expiry,
                "rv_long": episode.selection.rv_long,
                "rv_short": episode.selection.rv_short,
                "iv": episode.selection.iv,
                "sigma": episode.selection.sigma,
                "spot": episode.selection.spot,
                "strike": episode.selection.strike,
                "bid_up": episode.selection.bid_up,
                "ask_up": episode.selection.ask_up,
                "bid_down": episode.selection.bid_down,
                "ask_down": episode.selection.ask_down,
                "spread": episode.selection.spread,
                "fill_price": episode.fill_price,
                "fill_size": episode.fill_size,
                "order_notional": episode.order_notional,
                "settlement_outcome": record.settlement_outcome,
                "pnl": pnl,
                "reward": reward,
                "risk_limit_breach": risk_limit_breach,
            }
        )
        return reward

    def _choose_profile(self, *, context_id: str, allow_exploration: bool) -> tuple[str, bool]:
        profile_ids = list(self._profiles.keys())
        if not allow_exploration:
            return self._best_profile_no_exploration(context_id=context_id), False

        epsilon = min(1.0, max(0.0, self._config.exploration_epsilon))
        if epsilon > 0.0 and self._random.random() < epsilon:
            return str(self._random.choice(profile_ids)), True

        return self._best_profile_ucb(context_id=context_id), False

    def _best_profile_no_exploration(self, *, context_id: str) -> str:
        context_stats = self._stats.get(context_id, {})
        best_id = self._baseline_profile_id
        best_mean = float("-inf")
        for profile_id in self._profiles:
            stats = context_stats.get(profile_id)
            if stats is None or stats.count <= 0:
                continue
            if stats.mean_reward > best_mean:
                best_mean = stats.mean_reward
                best_id = profile_id
        return best_id

    def _best_profile_ucb(self, *, context_id: str) -> str:
        context_stats = self._stats.get(context_id, {})
        total = sum(stats.count for stats in context_stats.values()) + 1
        best_score = float("-inf")
        best_id = self._baseline_profile_id
        for profile_id in self._profiles:
            stats = context_stats.get(profile_id)
            if stats is None or stats.count <= 0:
                return profile_id
            bonus = self._config.ucb_c * sqrt(log(float(total)) / float(stats.count))
            score = stats.mean_reward + bonus
            if score > best_score:
                best_score = score
                best_id = profile_id
        return best_id

    def _update_stats(self, *, context_id: str, profile_id: str, reward: float) -> None:
        context_stats = self._stats.setdefault(context_id, {})
        arm = context_stats.setdefault(profile_id, _BanditArmStats())
        arm.count += 1
        arm.total_reward += reward

def _default_profiles() -> dict[str, PolicyProfile]:
    return {
        "P0_NOTRADE": PolicyProfile(
            profile_id="P0_NOTRADE",
            force_no_trade=True,
        ),
        "P1_CONSERVATIVE": PolicyProfile(
            profile_id="P1_CONSERVATIVE",
            edge_min_delta=0.010,
            edge_buffer_delta=0.005,
            cost_rate_delta=0.001,
            kelly_fraction_multiplier=0.60,
        ),
        "P2_BALANCED": PolicyProfile(profile_id="P2_BALANCED"),
        "P3_AGGRESSIVE": PolicyProfile(
            profile_id="P3_AGGRESSIVE",
            edge_min_delta=-0.005,
            edge_buffer_delta=-0.003,
            cost_rate_delta=0.002,
            kelly_fraction_multiplier=1.35,
        ),
        "P4_WIDE_SPREAD_GUARD": PolicyProfile(
            profile_id="P4_WIDE_SPREAD_GUARD",
            edge_min_delta=0.020,
            edge_buffer_delta=0.010,
            cost_rate_delta=0.002,
            kelly_fraction_multiplier=0.35,
        ),
    }


def build_context_id(
    *,
    seconds_to_expiry: float,
    rv_long: float,
    rv_short: float,
    max_spread: float,
    vol_ratio_threshold: float,
    spread_tight_threshold: float,
) -> str:
    if seconds_to_expiry < 15.0 * 60.0:
        tte_bucket = "tte_lt_15m"
    elif seconds_to_expiry <= 45.0 * 60.0:
        tte_bucket = "tte_15_45m"
    else:
        tte_bucket = "tte_gt_45m"

    base = max(rv_long, 1e-9)
    vol_ratio = max(rv_short, 0.0) / base
    vol_bucket = "vol_high" if vol_ratio >= max(1.0, vol_ratio_threshold) else "vol_low"
    spread_bucket = (
        "spread_tight"
        if max(max_spread, 0.0) <= max(0.0, spread_tight_threshold)
        else "spread_wide"
    )
    return f"{tte_bucket}|{vol_bucket}|{spread_bucket}"


def _serialize_row(row: dict[str, object]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            serialized[key] = value.isoformat()
            continue
        serialized[key] = value
    return serialized


def _serialize_selection(selection: PolicySelection) -> dict[str, object]:
    return {
        "timestamp": selection.timestamp.isoformat(),
        "market_id": selection.market_id,
        "context_id": selection.context_id,
        "chosen_profile_id": selection.chosen_profile_id,
        "applied_profile_id": selection.applied_profile_id,
        "exploration": selection.exploration,
        "bankroll_at_entry": selection.bankroll_at_entry,
        "seconds_to_expiry": selection.seconds_to_expiry,
        "rv_long": selection.rv_long,
        "rv_short": selection.rv_short,
        "iv": selection.iv,
        "sigma": selection.sigma,
        "spot": selection.spot,
        "strike": selection.strike,
        "bid_up": selection.bid_up,
        "ask_up": selection.ask_up,
        "bid_down": selection.bid_down,
        "ask_down": selection.ask_down,
        "spread": selection.spread,
    }


def _deserialize_selection(raw: object) -> PolicySelection | None:
    if not isinstance(raw, Mapping):
        return None
    timestamp = _parse_datetime(raw.get("timestamp"))
    if timestamp is None:
        return None
    market_id = str(raw.get("market_id") or "").strip()
    context_id = str(raw.get("context_id") or "").strip()
    chosen_profile_id = str(raw.get("chosen_profile_id") or "").strip()
    applied_profile_id = str(raw.get("applied_profile_id") or "").strip()
    exploration = bool(raw.get("exploration", False))
    bankroll_at_entry = _safe_float(raw.get("bankroll_at_entry"))
    seconds_to_expiry = _safe_float(raw.get("seconds_to_expiry"))
    rv_long = _safe_float(raw.get("rv_long"))
    rv_short = _safe_float(raw.get("rv_short"))
    sigma = _safe_float(raw.get("sigma"))
    spot = _safe_float(raw.get("spot"))
    strike = _safe_float(raw.get("strike"))
    bid_up = _safe_float(raw.get("bid_up"))
    ask_up = _safe_float(raw.get("ask_up"))
    bid_down = _safe_float(raw.get("bid_down"))
    ask_down = _safe_float(raw.get("ask_down"))
    spread = _safe_float(raw.get("spread"))
    if (
        not market_id
        or not context_id
        or not chosen_profile_id
        or not applied_profile_id
        or bankroll_at_entry is None
        or seconds_to_expiry is None
        or rv_long is None
        or rv_short is None
        or sigma is None
        or spot is None
        or strike is None
        or bid_up is None
        or ask_up is None
        or bid_down is None
        or ask_down is None
        or spread is None
    ):
        return None
    iv_raw = raw.get("iv")
    iv = _safe_float(iv_raw) if iv_raw is not None else None
    return PolicySelection(
        timestamp=timestamp,
        market_id=market_id,
        context_id=context_id,
        chosen_profile_id=chosen_profile_id,
        applied_profile_id=applied_profile_id,
        exploration=exploration,
        bankroll_at_entry=bankroll_at_entry,
        seconds_to_expiry=seconds_to_expiry,
        rv_long=rv_long,
        rv_short=rv_short,
        iv=iv,
        sigma=sigma,
        spot=spot,
        strike=strike,
        bid_up=bid_up,
        ask_up=ask_up,
        bid_down=bid_down,
        ask_down=ask_down,
        spread=spread,
    )


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None

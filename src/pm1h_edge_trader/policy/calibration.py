from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Sequence


def build_calibrator_features(
    *,
    q_model_up: float,
    spread: float,
    rv_long: float,
    rv_short: float,
    tte_seconds: float,
) -> tuple[float, float, float, float]:
    q = min(1.0 - 1e-6, max(1e-6, float(q_model_up)))
    logit_q = log(q / (1.0 - q))
    spread_f = max(0.0, float(spread))
    base = max(1e-9, float(rv_long))
    vol_ratio = max(0.0, float(rv_short)) / base
    tte_norm = min(1.0, max(0.0, float(tte_seconds) / 3600.0))
    return (logit_q, spread_f, vol_ratio, tte_norm)


@dataclass(frozen=True)
class OnlineCalibratorConfig:
    learning_rate: float = 1e-3
    l2: float = 1e-4
    max_weight_norm: float = 5.0
    feature_schema_version: int = 1


class OnlineLogisticCalibrator:
    """Tiny online logistic calibrator for q_model -> q_cal."""

    def __init__(
        self,
        *,
        learning_rate: float = 1e-3,
        l2: float = 1e-4,
        max_weight_norm: float = 5.0,
        feature_schema_version: int = 1,
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
        step_count: int = 0,
    ) -> None:
        self._config = OnlineCalibratorConfig(
            learning_rate=max(0.0, float(learning_rate)),
            l2=max(0.0, float(l2)),
            max_weight_norm=max(1e-9, float(max_weight_norm)),
            feature_schema_version=max(1, int(feature_schema_version)),
        )
        self._weights = [float(item) for item in (weights or (0.0, 0.0, 0.0, 0.0))]
        if len(self._weights) != 4:
            self._weights = [0.0, 0.0, 0.0, 0.0]
        self._bias = float(bias)
        self._step_count = max(0, int(step_count))

    def predict(self, features: Sequence[float]) -> float:
        x = _normalize_features(features)
        z = self._bias
        for idx, value in enumerate(x):
            z += self._weights[idx] * value
        return _sigmoid(z)

    def update(
        self,
        *,
        features: Sequence[float],
        label_up: float,
        freeze: bool = False,
    ) -> float:
        x = _normalize_features(features)
        y = 1.0 if float(label_up) >= 0.5 else 0.0
        p = self.predict(x)
        if freeze:
            return p

        error = p - y
        lr = self._config.learning_rate
        l2 = self._config.l2
        for idx, value in enumerate(x):
            grad = (error * value) + (l2 * self._weights[idx])
            self._weights[idx] -= lr * grad
        self._bias -= lr * error
        self._clip_weight_norm()
        self._step_count += 1
        return p

    def export_state(self) -> dict[str, float | int | list[float]]:
        return {
            "weights": list(self._weights),
            "bias": self._bias,
            "step_count": self._step_count,
            "lr": self._config.learning_rate,
            "l2": self._config.l2,
            "max_weight_norm": self._config.max_weight_norm,
            "feature_schema_version": self._config.feature_schema_version,
        }

    def import_state(self, payload: dict[str, object]) -> None:
        weights = payload.get("weights")
        if isinstance(weights, list) and len(weights) == 4:
            parsed: list[float] = []
            for value in weights:
                if isinstance(value, (int, float)):
                    parsed.append(float(value))
                else:
                    break
            if len(parsed) == 4:
                self._weights = parsed
        bias = payload.get("bias")
        if isinstance(bias, (int, float)):
            self._bias = float(bias)
        step_count = payload.get("step_count")
        if isinstance(step_count, int):
            self._step_count = max(0, step_count)

    def _clip_weight_norm(self) -> None:
        norm_sq = sum(value * value for value in self._weights)
        norm = sqrt(max(norm_sq, 0.0))
        cap = self._config.max_weight_norm
        if norm <= cap:
            return
        scale = cap / max(norm, 1e-12)
        self._weights = [value * scale for value in self._weights]


def _normalize_features(values: Sequence[float]) -> tuple[float, float, float, float]:
    safe = [0.0, 0.0, 0.0, 0.0]
    for idx in range(min(4, len(values))):
        safe[idx] = float(values[idx])
    return (safe[0], safe[1], safe[2], safe[3])


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


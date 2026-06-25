"""AI Forex Decision Agent — Central Configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Decision(str, Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    WAIT = "Wait"
    NO_TRADE = "No Trade"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


class AIModel(str, Enum):
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"


class Timeframe(str, Enum):
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"


class Session(str, Enum):
    SYDNEY = "Sydney"
    TOKYO = "Tokyo"
    LONDON = "London"
    NEW_YORK = "New York"


@dataclass(frozen=True)
class IndicatorWeights:
    """Weight configuration for each indicator component (total = 100)."""

    ema: int = 15
    macd: int = 10
    rsi: int = 10
    atr: int = 5
    adx: int = 10
    trend: int = 15
    support_resistance: int = 10
    candlestick: int = 10
    correlation: int = 20
    news: int = 5

    @property
    def total(self) -> int:
        return (
            self.ema + self.macd + self.rsi + self.atr + self.adx
            + self.trend + self.support_resistance + self.candlestick
            + self.correlation + self.news
        )


@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters."""

    risk_pct: float = 0.17
    reward_pct: float = 0.34
    risk_reward_ratio: float = 2.0


@dataclass(frozen=True)
class NewsConfig:
    """News filter parameters."""

    high_impact_minutes: int = 30


# Positive correlation pairs
POSITIVE_CORRELATION: dict[str, list[str]] = {
    "EURUSD": ["GBPUSD"],
    "GBPUSD": ["EURUSD"],
    "AUDUSD": ["NZDUSD"],
    "NZDUSD": ["AUDUSD"],
    "EURCAD": ["GBPCAD"],
    "GBPCAD": ["EURCAD"],
    "USDCAD": ["USDCHF"],
    "USDCHF": ["USDCAD"],
    "EURNZD": ["GBPNZD"],
    "GBPNZD": ["EURNZD"],
    "EURAUD": ["GBPAUD"],
    "GBPAUD": ["EURAUD"],
    "EURUSD": ["AUDUSD"],
    "AUDUSD": ["EURUSD"],
}

# Negative correlation pairs
NEGATIVE_CORRELATION: dict[str, list[str]] = {
    "AUDUSD": ["USDCHF"],
    "USDCHF": ["AUDUSD"],
    "EURUSD": ["USDCHF"],
    "USDCHF": ["EURUSD"],
    "EURUSD": ["USDCAD"],
    "USDCAD": ["EURUSD"],
    "NZDUSD": ["USDCHF"],
    "USDCHF": ["NZDUSD"],
    "GBPUSD": ["USDCHF"],
    "USDCHF": ["GBPUSD"],
    "GBPUSD": ["USDCAD"],
    "USDCAD": ["GBPUSD"],
}

# Full correlation map: main pair -> {positive: [...], negative: [...]}
CORRELATION_MAP: dict[str, dict[str, list[str]]] = {}

for _pair in set(list(POSITIVE_CORRELATION.keys()) + list(NEGATIVE_CORRELATION.keys())):
    CORRELATION_MAP[_pair] = {
        "positive": list(set(POSITIVE_CORRELATION.get(_pair, []))),
        "negative": list(set(NEGATIVE_CORRELATION.get(_pair, []))),
    }

# Remove duplicates within the same pair's positive/negative lists
for _p, _m in CORRELATION_MAP.items():
    _m["positive"] = list(set(_m["positive"]))
    _m["negative"] = list(set(_m["negative"]))
    # Remove self
    _m["positive"] = [x for x in _m["positive"] if x != _p]
    _m["negative"] = [x for x in _m["negative"] if x != _p]

# All available pairs
ALL_PAIRS: list[str] = sorted(set(list(CORRELATION_MAP.keys()) + [
    item for sublist in POSITIVE_CORRELATION.values() for item in sublist
] + [item for sublist in NEGATIVE_CORRELATION.values() for item in sublist]))


@dataclass
class AppConfig:
    """Top-level application configuration."""

    weights: IndicatorWeights = field(default_factory=IndicatorWeights)
    risk: RiskConfig = field(default_factory=RiskConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    ai_model: AIModel = AIModel.GPT4O_MINI
    default_timeframe: Timeframe = Timeframe.H1


def get_correlation_pairs(pair: str) -> dict[str, list[str]]:
    """Return correlated pairs for a given forex pair."""
    return CORRELATION_MAP.get(pair, {"positive": [], "negative": []})


def score_to_decision(score: float) -> Decision:
    """Convert a numeric score (0-100) to a Decision enum."""
    if score >= 90:
        return Decision.STRONG_BUY
    if score >= 80:
        return Decision.BUY
    if score >= 60:
        return Decision.WAIT
    if score >= 40:
        return Decision.NO_TRADE
    if score >= 20:
        return Decision.SELL
    return Decision.STRONG_SELL
"""Risk management utilities — entry, SL, TP, R:R calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import Decision, RiskConfig


@dataclass
class RiskResult:
    """Container for risk management calculations."""

    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    risk_pips: float
    reward_pips: float
    lot_size: float
    risk_amount: float
    reward_amount: float


def calculate_risk(
    current_price: float,
    atr_value: float,
    decision: Decision,
    risk_config: Optional[RiskConfig] = None,
    balance: float = 10000.0,
) -> RiskResult:
    """Calculate entry, stop-loss, take-profit, and position sizing.

    Args:
        current_price: Current market price.
        atr_value: ATR(14) value for dynamic SL/TP sizing.
        decision: The trade direction decision.
        risk_config: Risk management parameters. Defaults to class defaults.
        balance: Account balance in base currency.

    Returns:
        A ``RiskResult`` with all computed values.
    """
    if risk_config is None:
        risk_config = RiskConfig()

    risk_pips = atr_value * 1.5
    reward_pips = risk_pips * risk_config.risk_reward_ratio

    if decision in (Decision.STRONG_BUY, Decision.BUY):
        entry = current_price
        stop_loss = round(current_price - risk_pips, 5)
        take_profit = round(current_price + reward_pips, 5)
    elif decision in (Decision.STRONG_SELL, Decision.SELL):
        entry = current_price
        stop_loss = round(current_price + risk_pips, 5)
        take_profit = round(current_price - reward_pips, 5)
    else:
        entry = current_price
        stop_loss = current_price
        take_profit = current_price
        risk_pips = 0.0
        reward_pips = 0.0

    risk_amount = balance * (risk_config.risk_pct / 100.0)
    reward_amount = balance * (risk_config.reward_pct / 100.0)

    lot_size = 0.0
    if risk_pips > 0:
        pip_value = risk_pips * 100000  # Standard lot pip value approximation
        if pip_value > 0:
            lot_size = round(risk_amount / pip_value, 2)

    return RiskResult(
        entry=round(entry, 5),
        stop_loss=round(stop_loss, 5),
        take_profit=round(take_profit, 5),
        risk_reward=risk_config.risk_reward_ratio,
        risk_pips=round(risk_pips, 5),
        reward_pips=round(reward_pips, 5),
        lot_size=lot_size,
        risk_amount=round(risk_amount, 2),
        reward_amount=round(reward_amount, 2),
    )
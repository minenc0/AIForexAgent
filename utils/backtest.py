"""Backtesting engine — performance metrics from trade history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from utils.logger import logger


@dataclass
class BacktestResult:
    """Container for backtest performance metrics."""

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    no_trades: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0


def compute_backtest(trades: list[dict[str, Any]]) -> BacktestResult:
    """Calculate backtest metrics from a list of trade records.

    Args:
        trades: List of trade dictionaries, each containing at least
            ``outcome`` and ``profit_loss`` keys.

    Returns:
        A ``BacktestResult`` with all computed metrics.
    """
    result = BacktestResult()

    if not trades:
        logger.warning("No trades provided for backtest")
        return result

    resolved = [t for t in trades if t.get("outcome") in ("win", "loss")]
    no_trades = [t for t in trades if t.get("outcome") == "no_trade"]

    result.total_trades = len(trades)
    result.no_trades = len(no_trades)
    result.wins = sum(1 for t in resolved if t["outcome"] == "win")
    result.losses = sum(1 for t in resolved if t["outcome"] == "loss")

    total_resolved = result.wins + result.losses
    if total_resolved > 0:
        result.win_rate = round((result.wins / total_resolved) * 100, 2)
        result.loss_rate = round((result.losses / total_resolved) * 100, 2)

    profits = [t["profit_loss"] for t in resolved if t["outcome"] == "win" and t["profit_loss"]]
    losses = [t["profit_loss"] for t in resolved if t["outcome"] == "loss" and t["profit_loss"]]

    result.total_profit = sum(profits) if profits else 0.0
    result.total_loss = abs(sum(losses)) if losses else 0.0
    result.avg_profit = np.mean(profits) if profits else 0.0
    result.avg_loss = np.mean(losses) if losses else 0.0
    result.best_trade = max(profits) if profits else 0.0
    result.worst_trade = min(losses) if losses else 0.0

    if result.total_loss > 0:
        result.profit_factor = round(result.total_profit / result.total_loss, 2)
    elif result.total_profit > 0:
        result.profit_factor = float("inf")

    # Max drawdown from equity curve
    all_pnl = [t.get("profit_loss", 0.0) for t in resolved if t.get("profit_loss")]
    if all_pnl:
        cumulative = np.cumsum(all_pnl)
        peak = np.maximum.accumulate(cumulative)
        drawdown = cumulative - peak
        result.max_drawdown = round(float(np.min(drawdown)), 5)

    # Sharpe ratio (simplified, assuming risk-free = 0)
    if len(all_pnl) > 1:
        returns = np.diff(all_pnl)
        std = np.std(returns)
        if std > 0:
            result.sharpe_ratio = round(float(np.mean(returns) / std), 2)

    logger.info(
        "Backtest complete: %d trades, WR=%.1f%%, PF=%.2f",
        result.total_trades,
        result.win_rate,
        result.profit_factor,
    )
    return result
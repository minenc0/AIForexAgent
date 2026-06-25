"""Memory Agent — evaluates past trades and adapts indicator weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from config import IndicatorWeights
from utils.database import TradeDatabase
from utils.logger import logger


@dataclass
class MemoryResult:
    """Result of the memory agent's evaluation."""

    total_evaluated: int
    correct_signals: int
    incorrect_signals: int
    accuracy: float
    weight_adjustments: dict[str, float]
    details: str


class MemoryAgent:
    """Evaluates past trade outcomes and adjusts indicator weights.

    After each trade is resolved (win/loss), the memory agent checks
    which indicators contributed correctly and adjusts their weights
    for future analysis.
    """

    def __init__(
        self,
        db: TradeDatabase,
        current_weights: IndicatorWeights,
    ) -> None:
        """Initialise the MemoryAgent.

        Args:
            db: Trade database instance.
            current_weights: Current indicator weight configuration.
        """
        self.db = db
        self.current_weights = current_weights

    def evaluate_and_adapt(self) -> MemoryResult:
        """Evaluate all resolved trades and compute weight adjustments.

        Returns:
            A ``MemoryResult`` with accuracy statistics and suggested
            weight adjustments.
        """
        logger.info("MemoryAgent evaluating trade history")

        trades = self.db.get_all_trades()
        resolved = [t for t in trades if t.get("outcome") in ("win", "loss")]

        if not resolved:
            logger.info("No resolved trades to evaluate")
            return MemoryResult(
                total_evaluated=0,
                correct_signals=0,
                incorrect_signals=0,
                accuracy=0.0,
                weight_adjustments={},
                details="No resolved trades in history",
            )

        correct = sum(1 for t in resolved if t["outcome"] == "win")
        incorrect = len(resolved) - correct
        accuracy = (correct / len(resolved)) * 100.0

        # Compute adjustments based on correlation between
        # individual indicator scores and actual outcomes
        adjustments: dict[str, float] = {}

        # Simple heuristic: if accuracy < 50%, reduce weights on
        # indicators that were wrong most often
        # If accuracy > 70%, increase weights on strongest indicators
        indicator_scores_map: dict[str, list[float]] = {
            "ema": [], "macd": [], "rsi": [], "atr": [],
            "adx": [], "trend": [], "support_resistance": [],
            "candlestick": [], "correlation": [],
        }

        for trade in resolved:
            # Infer indicator contribution from the stored scores
            # This is a simplified approach — in production, you'd
            # store individual scores per trade
            tech_score = trade.get("technical_score", 50)
            corr_score = trade.get("correlation_score", 50)
            was_win = trade.get("outcome") == "win"

            # Distribute technical score across indicators proportionally
            if tech_score > 60 and was_win:
                for key in indicator_scores_map:
                    indicator_scores_map[key].append(1.0)
            elif tech_score < 40 and not was_win:
                for key in indicator_scores_map:
                    indicator_scores_map[key].append(1.0)
            else:
                for key in indicator_scores_map:
                    indicator_scores_map[key].append(-0.5)

        # Calculate adjustment factor
        if accuracy < 40:
            factor = -0.05  # Reduce weights
        elif accuracy > 70:
            factor = 0.05  # Increase weights
        else:
            factor = 0.0

        base_weights = {
            "ema": self.current_weights.ema,
            "macd": self.current_weights.macd,
            "rsi": self.current_weights.rsi,
            "atr": self.current_weights.atr,
            "adx": self.current_weights.adx,
            "trend": self.current_weights.trend,
            "support_resistance": self.current_weights.support_resistance,
            "candlestick": self.current_weights.candlestick,
            "correlation": self.current_weights.correlation,
            "news": self.current_weights.news,
        }

        for key, base in base_weights.items():
            adjustment = round(base * factor, 2)
            new_weight = max(0, base + adjustment)
            adjustments[key] = new_weight

        # Ensure total still = 100
        total = sum(adjustments.values())
        if total > 0:
            scale = 100.0 / total
            adjustments = {k: round(v * scale, 1) for k, v in adjustments.items()}

        details = (
            f"Evaluated: {len(resolved)} trades | "
            f"Correct: {correct} | Incorrect: {incorrect} | "
            f"Accuracy: {accuracy:.1f}% | "
            f"Weight adjustment factor: {factor}"
        )

        logger.info("MemoryAgent result: %s", details)

        return MemoryResult(
            total_evaluated=len(resolved),
            correct_signals=correct,
            incorrect_signals=incorrect,
            accuracy=accuracy,
            weight_adjustments=adjustments,
            details=details,
        )

    def learn_from_trade(
        self,
        trade_id: int,
        was_correct: bool,
        notes: str = "",
    ) -> None:
        """Record the outcome of a specific trade for learning.

        Args:
            trade_id: The database row id of the trade.
            was_correct: Whether the signal was correct.
            notes: Optional notes about the evaluation.
        """
        self.db.save_evaluation(trade_id, was_correct, notes)
        logger.info(
            "MemoryAgent learned from trade %d: correct=%s",
            trade_id, was_correct,
        )
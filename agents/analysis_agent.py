"""Technical Analysis Agent — aggregates all indicator scores."""

from __future__ import annotations

from dataclasses import dataclass

from config import IndicatorWeights, Timeframe
from data.technical import compute_technical_analysis, TechnicalResult, compute_weighted_technical_score
from utils.logger import logger


@dataclass
class TechnicalAgentResult:
    """Result of the technical analysis agent."""

    technical_score: float  # 0-100
    individual_scores: dict[str, float]
    details: str
    raw: Optional[TechnicalResult] = None


class TechnicalAgent:
    """Aggregates scores from all technical indicators using configurable weights.

    Computes each indicator's score and combines them into a single
    technical score.
    """

    def __init__(self, weights: Optional[IndicatorWeights] = None) -> None:
        """Initialise the TechnicalAgent.

        Args:
            weights: Custom indicator weights. Uses defaults if ``None``.
        """
        self.weights = weights or IndicatorWeights()

    def analyse(
        self,
        pair: str,
        timeframe: Timeframe,
    ) -> TechnicalAgentResult:
        """Run the full technical analysis.

        Args:
            pair: Forex pair string.
            timeframe: Bar period.

        Returns:
            A ``TechnicalAgentResult`` with the aggregated score and
            individual indicator scores.
        """
        logger.info("TechnicalAgent analysing %s @ %s", pair, timeframe.value)

        try:
            ta = compute_technical_analysis(pair, timeframe)
        except (ValueError, Exception) as e:
            logger.error("TechnicalAgent: failed for %s @ %s: %s", pair, timeframe.value, e)
            return TechnicalAgentResult(
                technical_score=50.0,
                individual_scores={},
                details=f"Data unavailable: {e}",
            )

        individual_scores = {
            "EMA": ta.ema_score,
            "MACD": ta.macd_score,
            "RSI": ta.rsi_score,
            "ATR": ta.atr_score,
            "ADX": ta.adx_score,
            "Trend": ta.ema_score,  # Trend derived from EMA alignment
            "Support/Resistance": (ta.support_score + ta.resistance_score) / 2.0,
            "Candlestick": ta.candlestick_score,
        }

        weight_map = {
            "ema": self.weights.ema,
            "macd": self.weights.macd,
            "rsi": self.weights.rsi,
            "atr": self.weights.atr,
            "adx": self.weights.adx,
            "trend": self.weights.trend,
            "support_resistance": self.weights.support_resistance,
            "candlestick": self.weights.candlestick,
        }

        tech_score = compute_weighted_technical_score(ta, weight_map)

        details = (
            f"Technical Score: {tech_score:.1f} | "
            f"EMA: {ta.ema_score:.1f} MACD: {ta.macd_score:.1f} "
            f"RSI: {ta.rsi_score:.1f} ATR: {ta.atr_score:.1f} "
            f"ADX: {ta.adx_score:.1f} SR: {(ta.support_score + ta.resistance_score) / 2:.1f} "
            f"Candle: {ta.candlestick_score:.1f}"
        )

        logger.info("TechnicalAgent result: %s", details)

        return TechnicalAgentResult(
            technical_score=tech_score,
            individual_scores=individual_scores,
            details=details,
            raw=ta,
        )
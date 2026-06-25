"""Trend Analysis Agent — determines overall trend direction and strength."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import Timeframe
from data.market import get_market_data
from data.technical import compute_technical_analysis, TechnicalResult
from utils.logger import logger


@dataclass
class TrendResult:
    """Result of the trend analysis."""

    direction: str  # 'bullish', 'bearish', 'neutral'
    strength: str  # 'strong', 'moderate', 'weak'
    ema_alignment: str  # Description of EMA stack
    higher_tf_trend: str  # Trend from one timeframe up
    score: float  # 0-100
    details: str


class TrendAgent:
    """Analyses the trend using EMA alignment and multi-timeframe context.

    The trend agent examines EMA20, EMA50, and EMA200 stacking, price
    position relative to each EMA, and the trend on a higher timeframe
    for confirmation.
    """

    def analyse(
        self,
        pair: str,
        timeframe: Timeframe,
        use_higher_tf: bool = True,
    ) -> TrendResult:
        """Run the trend analysis.

        Args:
            pair: Forex pair string.
            timeframe: Current timeframe.
            use_higher_tf: Whether to check the higher timeframe for confirmation.

        Returns:
            A ``TrendResult`` with direction, strength, and score.
        """
        logger.info("TrendAgent analysing %s @ %s", pair, timeframe.value)

        result = TrendResult(
            direction="neutral",
            strength="weak",
            ema_alignment="No data",
            higher_tf_trend="N/A",
            score=50.0,
            details="",
        )

        # Current timeframe analysis
        try:
            ta = compute_technical_analysis(pair, timeframe)
        except (ValueError, Exception) as e:
            logger.error("TrendAgent: failed to get data for %s @ %s: %s", pair, timeframe.value, e)
            result.details = f"Data unavailable: {e}"
            return result

        price = ta.current_price
        e20 = ta.ema20
        e50 = ta.ema50
        e200 = ta.ema200

        # EMA Alignment
        if e20 is not None and e50 is not None and e200 is not None:
            if e20 > e50 > e200 and price > e20:
                result.direction = "bullish"
                result.ema_alignment = "Bullish Stack (EMA20 > EMA50 > EMA200, Price above all)"
                result.score = 85.0
                result.strength = "strong"
            elif e20 < e50 < e200 and price < e20:
                result.direction = "bearish"
                result.ema_alignment = "Bearish Stack (EMA20 < EMA50 < EMA200, Price below all)"
                result.score = 15.0
                result.strength = "strong"
            elif e20 > e50 and price > e50:
                result.direction = "bullish"
                result.ema_alignment = "Moderate Bullish (EMA20 > EMA50, Price above EMA50)"
                result.score = 65.0
                result.strength = "moderate"
            elif e20 < e50 and price < e50:
                result.direction = "bearish"
                result.ema_alignment = "Moderate Bearish (EMA20 < EMA50, Price below EMA50)"
                result.score = 35.0
                result.strength = "moderate"
            elif price > e200:
                result.direction = "bullish"
                result.ema_alignment = "Weak Bullish (Price above EMA200 only)"
                result.score = 58.0
                result.strength = "weak"
            elif price < e200:
                result.direction = "bearish"
                result.ema_alignment = "Weak Bearish (Price below EMA200 only)"
                result.score = 42.0
                result.strength = "weak"
            else:
                result.ema_alignment = "No clear EMA alignment"
                result.score = 50.0
        elif e20 is not None and e50 is not None:
            if e20 > e50 and price > e20:
                result.direction = "bullish"
                result.ema_alignment = "Bullish (EMA20 > EMA50, Price above)"
                result.score = 65.0
                result.strength = "moderate"
            elif e20 < e50 and price < e20:
                result.direction = "bearish"
                result.ema_alignment = "Bearish (EMA20 < EMA50, Price below)"
                result.score = 35.0
                result.strength = "moderate"
            else:
                result.ema_alignment = "Mixed EMA signals"
                result.score = 50.0
        else:
            result.ema_alignment = "Insufficient data for EMA analysis"
            result.score = 50.0

        # Higher timeframe confirmation
        if use_higher_tf:
            higher_tf = self._get_higher_timeframe(timeframe)
            if higher_tf:
                try:
                    higher_ta = compute_technical_analysis(pair, higher_tf)
                    higher_price = higher_ta.current_price
                    h_e20 = higher_ta.ema20
                    h_e50 = higher_ta.ema50

                    if h_e20 is not None and h_e50 is not None:
                        if h_e20 > h_e50 and higher_price > h_e20:
                            result.higher_tf_trend = "Bullish"
                            if result.direction == "bullish":
                                result.score = min(100, result.score + 5)
                        elif h_e20 < h_e50 and higher_price < h_e20:
                            result.higher_tf_trend = "Bearish"
                            if result.direction == "bearish":
                                result.score = max(0, result.score - 5)
                        else:
                            result.higher_tf_trend = "Mixed"
                            if result.direction != "neutral":
                                result.score = result.score * 0.9 + 50 * 0.1
                except (ValueError, Exception) as e:
                    logger.debug("Higher TF analysis failed: %s", e)
                    result.higher_tf_trend = "Unavailable"

        result.details = (
            f"Trend: {result.direction} ({result.strength}) | "
            f"EMA: {result.ema_alignment} | "
            f"Higher TF: {result.higher_tf_trend} | "
            f"Score: {result.score:.1f}"
        )

        logger.info("TrendAgent result: %s", result.details)
        return result

    @staticmethod
    def _get_higher_timeframe(tf: Timeframe) -> Optional[Timeframe]:
        """Return the next higher timeframe, or None for H4.

        Args:
            tf: Current timeframe.

        Returns:
            The next timeframe up, or ``None`` if already at H4.
        """
        hierarchy = [Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H4]
        try:
            idx = hierarchy.index(tf)
            if idx < len(hierarchy) - 1:
                return hierarchy[idx + 1]
        except ValueError:
            pass
        return None
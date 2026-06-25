"""Correlation Analysis Agent — validates signals across correlated pairs."""

from __future__ import annotations

from dataclasses import dataclass

from config import Timeframe
from data.correlation import run_correlation_analysis, CorrelationResult
from utils.logger import logger


@dataclass
class CorrelationAgentResult:
    """Result of the correlation agent analysis."""

    correlation_score: float  # 0-100
    confidence_modifier: float  # Can be negative
    checks: list[dict]
    details: str
    raw_result: Optional[CorrelationResult] = None


class CorrelationAgent:
    """Validates the main pair's signal by checking correlated pairs.

    If the main pair is bullish, positively correlated pairs should also
    be bullish and negatively correlated pairs should be bearish for
    confirmation.
    """

    def analyse(
        self,
        main_pair: str,
        main_direction: str,
        timeframe: Timeframe,
    ) -> CorrelationAgentResult:
        """Run the correlation analysis.

        Args:
            main_pair: The primary forex pair being analysed.
            main_direction: ``'bullish'``, ``'bearish'``, or ``'neutral'``.
            timeframe: Bar period.

        Returns:
            A ``CorrelationAgentResult`` with scores and details.
        """
        logger.info(
            "CorrelationAgent analysing %s (direction=%s) @ %s",
            main_pair, main_direction, timeframe.value,
        )

        corr_result = run_correlation_analysis(main_pair, main_direction, timeframe)

        checks = [
            {
                "pair": c.pair,
                "type": c.correlation_type,
                "direction": c.direction,
                "score": c.score,
                "confirms": c.confirms,
            }
            for c in corr_result.checks
        ]

        result = CorrelationAgentResult(
            correlation_score=corr_result.correlation_score,
            confidence_modifier=corr_result.confidence_modifier,
            checks=checks,
            details=corr_result.details,
            raw_result=corr_result,
        )

        logger.info(
            "CorrelationAgent result: score=%.1f modifier=%.1f | %s",
            result.correlation_score,
            result.confidence_modifier,
            result.details,
        )
        return result
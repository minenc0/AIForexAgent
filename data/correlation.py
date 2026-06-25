"""Correlation engine — validates signals across correlated forex pairs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from config import Timeframe, get_correlation_pairs
from data.market import get_market_data
from indicators.ema import ema_score as calc_ema_score
from indicators.rsi import rsi_score as calc_rsi_score
from indicators.macd import macd_score as calc_macd_score
from utils.logger import logger


@dataclass
class CorrelationCheck:
    """Result of a single correlated pair check."""

    pair: str
    correlation_type: str  # 'positive' or 'negative'
    direction: str  # 'bullish', 'bearish', 'neutral'
    score: float  # 0-100
    confirms: bool


@dataclass
class CorrelationResult:
    """Aggregate correlation analysis result."""

    main_pair: str = ""
    checks: list[CorrelationCheck] = field(default_factory=list)
    correlation_score: float = 50.0
    confidence_modifier: float = 0.0
    details: str = ""


def _determine_pair_direction(
    pair: str,
    timeframe: Timeframe,
) -> tuple[str, float]:
    """Determine the overall direction and strength of a pair.

    Uses a simplified EMA + RSI + MACD composite to determine if the
    pair is bullish, bearish, or neutral.

    Args:
        pair: Forex pair string.
        timeframe: Bar period.

    Returns:
        A tuple of ``(direction, strength)`` where direction is
        ``'bullish'``, ``'bearish'``, or ``'neutral'`` and strength
        is a float 0-100.
    """
    try:
        df = get_market_data(pair, timeframe)
    except (ValueError, Exception):
        return "neutral", 50.0

    if len(df) < 50:
        return "neutral", 50.0

    ema_s = calc_ema_score(df)
    rsi_s = calc_rsi_score(df)
    macd_s = calc_macd_score(df)

    composite = (ema_s + rsi_s + macd_s) / 3.0

    if composite >= 65:
        return "bullish", composite
    elif composite <= 35:
        return "bearish", composite
    else:
        return "neutral", composite


def run_correlation_analysis(
    main_pair: str,
    main_direction: str,
    timeframe: Timeframe,
) -> CorrelationResult:
    """Run correlation analysis across all correlated pairs.

    For the main pair, fetches all positively and negatively correlated
    pairs and checks if their directions confirm the main signal.

    Args:
        main_pair: The primary forex pair being analysed.
        main_direction: ``'bullish'``, ``'bearish'``, or ``'neutral'``.
        timeframe: Bar period.

    Returns:
        A ``CorrelationResult`` with all check results and an aggregate
        score.
    """
    logger.info(
        "Running correlation analysis for %s (direction=%s) @ %s",
        main_pair, main_direction, timeframe.value,
    )

    result = CorrelationResult(main_pair=main_pair)
    corr_map = get_correlation_pairs(main_pair)

    if not corr_map["positive"] and not corr_map["negative"]:
        logger.info("No correlated pairs found for %s", main_pair)
        result.correlation_score = 50.0
        result.details = "No correlated pairs configured"
        return result

    total_score = 0.0
    check_count = 0

    # Check positive correlation pairs
    for pair in corr_map["positive"]:
        direction, strength = _determine_pair_direction(pair, timeframe)

        confirms = False
        if main_direction == "bullish" and direction == "bullish":
            confirms = True
        elif main_direction == "bearish" and direction == "bearish":
            confirms = True
        elif main_direction == "neutral":
            confirms = True

        score = strength if confirms else (100.0 - strength)
        total_score += score
        check_count += 1

        result.checks.append(CorrelationCheck(
            pair=pair,
            correlation_type="positive",
            direction=direction,
            score=score,
            confirms=confirms,
        ))

    # Check negative correlation pairs
    for pair in corr_map["negative"]:
        direction, strength = _determine_pair_direction(pair, timeframe)

        confirms = False
        if main_direction == "bullish" and direction == "bearish":
            confirms = True
        elif main_direction == "bearish" and direction == "bullish":
            confirms = True
        elif main_direction == "neutral":
            confirms = True

        score = strength if confirms else (100.0 - strength)
        total_score += score
        check_count += 1

        result.checks.append(CorrelationCheck(
            pair=pair,
            correlation_type="negative",
            direction=direction,
            score=score,
            confirms=confirms,
        ))

    if check_count > 0:
        result.correlation_score = round(total_score / check_count, 2)
    else:
        result.correlation_score = 50.0

    # Calculate confidence modifier
    if check_count > 0:
        confirming = sum(1 for c in result.checks if c.confirms)
        confirmation_ratio = confirming / check_count

        if confirmation_ratio >= 0.8:
            result.confidence_modifier = 10.0
            result.details = f"Strong confirmation: {confirming}/{check_count} pairs confirm"
        elif confirmation_ratio >= 0.6:
            result.confidence_modifier = 5.0
            result.details = f"Moderate confirmation: {confirming}/{check_count} pairs confirm"
        elif confirmation_ratio >= 0.4:
            result.confidence_modifier = 0.0
            result.details = f"Weak confirmation: {confirming}/{check_count} pairs confirm"
        else:
            result.confidence_modifier = -15.0
            result.details = f"Contradictory: only {confirming}/{check_count} pairs confirm"

    logger.info(
        "Correlation result for %s: score=%.1f modifier=%.1f | %s",
        main_pair, result.correlation_score, result.confidence_modifier,
        result.details,
    )

    return result


def correlation_score_to_weighted(
    corr_result: CorrelationResult,
    weight: float = 20.0,
) -> float:
    """Convert the raw correlation score to a weighted contribution.

    Args:
        corr_result: Correlation analysis result.
        weight: The weight assigned to correlation (default 20).

    Returns:
        Weighted score contribution (0 to weight).
    """
    return (corr_result.correlation_score / 100.0) * weight
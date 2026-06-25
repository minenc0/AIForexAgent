"""Candlestick pattern recognition engine."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pandas import Series

from utils.logger import logger


# Pattern registry: name → detection function
_PATTERNS: dict[str, callable] = {}


def _register(name: str):
    """Decorator to register a pattern detection function."""
    def decorator(fn: callable) -> callable:
        _PATTERNS[name] = fn
        return fn
    return decorator


@_register("Bullish Engulfing")
def bullish_engulfing(df: pd.DataFrame) -> bool:
    """Detect a Bullish Engulfing pattern at the last bar.

    Requires the previous bar to be bearish (close < open) and the
    current bar to be bullish (close > open) with the current body
    fully engulfing the previous body.

    Args:
        df: OHLC DataFrame (needs at least 2 rows).

    Returns:
        ``True`` if the pattern is detected.
    """
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    prev_body = prev["Open"] - prev["Close"]
    curr_body = curr["Close"] - curr["Open"]
    return (
        prev_body < 0  # Previous bearish
        and curr_body > 0  # Current bullish
        and curr["Open"] <= prev["Close"]  # Opens below/at prev close
        and curr["Close"] >= prev["Open"]  # Closes above/at prev open
    )


@_register("Bearish Engulfing")
def bearish_engulfing(df: pd.DataFrame) -> bool:
    """Detect a Bearish Engulfing pattern at the last bar.

    Args:
        df: OHLC DataFrame (needs at least 2 rows).

    Returns:
        ``True`` if the pattern is detected.
    """
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    prev_body = prev["Close"] - prev["Open"]
    curr_body = curr["Open"] - curr["Close"]
    return (
        prev_body > 0  # Previous bullish
        and curr_body > 0  # Current bearish
        and curr["Open"] >= prev["Close"]  # Opens above/at prev close
        and curr["Close"] <= prev["Open"]  # Closes below/at prev open
    )


@_register("Morning Star")
def morning_star(df: pd.DataFrame) -> bool:
    """Detect a Morning Star (3-candle bullish reversal) pattern.

    Args:
        df: OHLC DataFrame (needs at least 3 rows).

    Returns:
        ``True`` if the pattern is detected.
    """
    if len(df) < 3:
        return False
    first = df.iloc[-3]
    second = df.iloc[-2]
    third = df.iloc[-1]

    first_bearish = first["Close"] < first["Open"]
    small_body = abs(second["Close"] - second["Open"]) < (first["Open"] - first["Close"]) * 0.3
    third_bullish = third["Close"] > third["Open"]
    third_closes_above_mid = third["Close"] > (first["Open"] + first["Close"]) / 2

    return first_bearish and small_body and third_bullish and third_closes_above_mid


@_register("Evening Star")
def evening_star(df: pd.DataFrame) -> bool:
    """Detect an Evening Star (3-candle bearish reversal) pattern.

    Args:
        df: OHLC DataFrame (needs at least 3 rows).

    Returns:
        ``True`` if the pattern is detected.
    """
    if len(df) < 3:
        return False
    first = df.iloc[-3]
    second = df.iloc[-2]
    third = df.iloc[-1]

    first_bullish = first["Close"] > first["Open"]
    small_body = abs(second["Close"] - second["Open"]) < (first["Close"] - first["Open"]) * 0.3
    third_bearish = third["Close"] < third["Open"]
    third_closes_below_mid = third["Close"] < (first["Open"] + first["Close"]) / 2

    return first_bullish and small_body and third_bearish and third_closes_below_mid


@_register("Hammer")
def hammer(df: pd.DataFrame) -> bool:
    """Detect a Hammer pattern at the last bar.

    A candle with a small body at the top and a long lower shadow
    (at least 2x the body), appearing in a downtrend.

    Args:
        df: OHLC DataFrame (needs at least 5 rows).

    Returns:
        ``True`` if the pattern is detected.
    """
    if len(df) < 5:
        return False
    curr = df.iloc[-1]
    body = abs(curr["Close"] - curr["Open"])
    lower_shadow = min(curr["Open"], curr["Close"]) - curr["Low"]
    upper_shadow = curr["High"] - max(curr["Open"], curr["Close"])

    downtrend = df["Close"].iloc[-5:-1].is_monotonic_decreasing
    valid_shadow = lower_shadow >= body * 2
    small_upper = upper_shadow <= body * 0.5 if body > 0 else upper_shadow == 0

    return downtrend and valid_shadow and small_upper


@_register("Shooting Star")
def shooting_star(df: pd.DataFrame) -> bool:
    """Detect a Shooting Star pattern at the last bar.

    A candle with a small body at the bottom and a long upper shadow
    (at least 2x the body), appearing in an uptrend.

    Args:
        df: OHLC DataFrame (needs at least 5 rows).

    Returns:
        ``True`` if the pattern is detected.
    """
    if len(df) < 5:
        return False
    curr = df.iloc[-1]
    body = abs(curr["Close"] - curr["Open"])
    upper_shadow = curr["High"] - max(curr["Open"], curr["Close"])
    lower_shadow = min(curr["Open"], curr["Close"]) - curr["Low"]

    uptrend = df["Close"].iloc[-5:-1].is_monotonic_increasing
    valid_shadow = upper_shadow >= body * 2
    small_lower = lower_shadow <= body * 0.5 if body > 0 else lower_shadow == 0

    return uptrend and valid_shadow and small_lower


@_register("Doji")
def doji(df: pd.DataFrame) -> bool:
    """Detect a Doji pattern at the last bar.

    A candle where the open and close are virtually identical
    (body < 5% of the total range).

    Args:
        df: OHLC DataFrame (needs at least 1 row).

    Returns:
        ``True`` if the pattern is detected.
    """
    if len(df) < 1:
        return False
    curr = df.iloc[-1]
    total_range = curr["High"] - curr["Low"]
    if total_range == 0:
        return False
    body = abs(curr["Close"] - curr["Open"])
    return body < total_range * 0.05


@_register("Pin Bar")
def pin_bar(df: pd.DataFrame) -> bool:
    """Detect a Pin Bar pattern at the last bar.

    A candle with a long tail/wick and a very small body, where the
    tail is at least 2/3 of the total range.

    Args:
        df: OHLC DataFrame (needs at least 1 row).

    Returns:
        ``True`` if the pattern is detected.
    """
    if len(df) < 1:
        return False
    curr = df.iloc[-1]
    body = abs(curr["Close"] - curr["Open"])
    total_range = curr["High"] - curr["Low"]
    upper_shadow = curr["High"] - max(curr["Open"], curr["Close"])
    lower_shadow = min(curr["Open"], curr["Close"]) - curr["Low"]

    if total_range == 0:
        return False
    return (
        body < total_range * 0.25
        and (upper_shadow > total_range * 0.6 or lower_shadow > total_range * 0.6)
    )


@_register("Inside Bar")
def inside_bar(df: pd.DataFrame) -> bool:
    """Detect an Inside Bar pattern at the last bar.

    The current bar's entire range is within the previous bar's range.

    Args:
        df: OHLC DataFrame (needs at least 2 rows).

    Returns:
        ``True`` if the pattern is detected.
    """
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (
        curr["High"] <= prev["High"]
        and curr["Low"] >= prev["Low"]
    )


@_register("Outside Bar")
def outside_bar(df: pd.DataFrame) -> bool:
    """Detect an Outside Bar (Engulfing Bar) pattern at the last bar.

    The current bar's range completely engulfs the previous bar's range.

    Args:
        df: OHLC DataFrame (needs at least 2 rows).

    Returns:
        ``True`` if the pattern is detected.
    """
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (
        curr["High"] >= prev["High"]
        and curr["Low"] <= prev["Low"]
    )


def detect_all_patterns(df: pd.DataFrame) -> dict[str, bool]:
    """Run all registered pattern detectors on the DataFrame.

    Args:
        df: OHLC DataFrame.

    Returns:
        A dictionary mapping pattern name to ``True``/``False``.
    """
    results: dict[str, bool] = {}
    for name, fn in _PATTERNS.items():
        try:
            results[name] = fn(df)
        except Exception as e:
            logger.warning("Pattern detection error for %s: %s", name, e)
            results[name] = False
    return results


def candlestick_score(df: pd.DataFrame) -> float:
    """Compute a candlestick-based score from 0 to 100.

    Bullish patterns add points, bearish patterns subtract points.

    Args:
        df: OHLC DataFrame.

    Returns:
        A float between 0 and 100.
    """
    patterns = detect_all_patterns(df)
    score = 50.0

    bullish_patterns = {
        "Bullish Engulfing": 20,
        "Morning Star": 25,
        "Hammer": 20,
        "Pin Bar": 10,
        "Inside Bar": 5,
    }
    bearish_patterns = {
        "Bearish Engulfing": -20,
        "Evening Star": -25,
        "Shooting Star": -20,
        "Outside Bar": -10,
    }

    for name, weight in bullish_patterns.items():
        if patterns.get(name, False):
            score += weight

    for name, weight in bearish_patterns.items():
        if patterns.get(name, False):
            score += weight

    if patterns.get("Doji", False):
        score = score * 0.8  # Doji = indecision, reduce confidence

    return max(0.0, min(100.0, score))


def get_detected_patterns(df: pd.DataFrame) -> list[str]:
    """Return a list of detected pattern names.

    Args:
        df: OHLC DataFrame.

    Returns:
        List of pattern name strings that were detected.
    """
    patterns = detect_all_patterns(df)
    return [name for name, detected in patterns.items() if detected]
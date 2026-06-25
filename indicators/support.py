"""Support level detection using pivot points and swing lows."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from utils.logger import logger


def swing_lows(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Identify swing low points in the price series.

    A swing low is a bar where the Low is the lowest within the
    surrounding ``window`` bars on both sides.

    Args:
        df: OHLC DataFrame with a ``Low`` column.
        window: Number of bars on each side to check.

    Returns:
        A boolean Series where ``True`` marks a swing low.
    """
    lows = df["Low"]
    result = pd.Series(False, index=df.index)

    for i in range(window, len(df) - window):
        current = lows.iloc[i]
        left_min = lows.iloc[i - window : i].min()
        right_min = lows.iloc[i + 1 : i + window + 1].min()
        if current <= left_min and current <= right_min:
            result.iloc[i] = True

    return result


def find_support_levels(
    df: pd.DataFrame,
    current_price: float,
    num_levels: int = 3,
    window: int = 10,
) -> list[float]:
    """Find the nearest support levels below the current price.

    Uses swing low detection and returns the most recent, relevant
    support levels.

    Args:
        df: OHLC DataFrame.
        current_price: Current market price.
        num_levels: Maximum number of support levels to return.
        window: Swing detection window.

    Returns:
        A sorted list of support prices (lowest first).
    """
    if len(df) < window * 2:
        return []

    swings = swing_lows(df, window)
    swing_prices = df.loc[swings, "Low"].tolist()

    # Filter: only levels below current price
    supports = [s for s in swing_prices if s < current_price]

    # Remove duplicates (within 0.0001 tolerance)
    unique: list[float] = []
    for s in sorted(supports, reverse=True):  # Most recent first
        if not any(abs(s - u) < 0.0001 for u in unique):
            unique.append(s)

    # Return the closest num_levels below current price
    closest = sorted(unique, reverse=True)[:num_levels]
    logger.debug("Support levels found: %s", closest)
    return sorted(closest)


def support_score(
    df: pd.DataFrame,
    current_price: float,
) -> float:
    """Score based on proximity to the nearest support level.

    Closer to support → more bullish (buy near support).
    If price is far from any support → neutral.

    Args:
        df: OHLC DataFrame.
        current_price: Current market price.

    Returns:
        A float between 0 and 100.
    """
    supports = find_support_levels(df, current_price, num_levels=1)
    if not supports:
        return 50.0

    nearest = supports[-1]  # Highest support below price
    distance = current_price - nearest

    # Use ATR to normalise distance
    from indicators.atr import atr14

    atr_val = atr14(df).iloc[-1]
    if pd.isna(atr_val) or atr_val == 0:
        return 50.0

    ratio = distance / atr_val

    if ratio <= 0.5:
        return 85.0  # Very close to support
    elif ratio <= 1.0:
        return 70.0
    elif ratio <= 2.0:
        return 55.0
    else:
        return 50.0
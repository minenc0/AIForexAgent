"""Resistance level detection using pivot points and swing highs."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from utils.logger import logger


def swing_highs(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Identify swing high points in the price series.

    A swing high is a bar where the High is the highest within the
    surrounding ``window`` bars on both sides.

    Args:
        df: OHLC DataFrame with a ``High`` column.
        window: Number of bars on each side to check.

    Returns:
        A boolean Series where ``True`` marks a swing high.
    """
    highs = df["High"]
    result = pd.Series(False, index=df.index)

    for i in range(window, len(df) - window):
        current = highs.iloc[i]
        left_max = highs.iloc[i - window : i].max()
        right_max = highs.iloc[i + 1 : i + window + 1].max()
        if current >= left_max and current >= right_max:
            result.iloc[i] = True

    return result


def find_resistance_levels(
    df: pd.DataFrame,
    current_price: float,
    num_levels: int = 3,
    window: int = 10,
) -> list[float]:
    """Find the nearest resistance levels above the current price.

    Uses swing high detection and returns the most recent, relevant
    resistance levels.

    Args:
        df: OHLC DataFrame.
        current_price: Current market price.
        num_levels: Maximum number of resistance levels to return.
        window: Swing detection window.

    Returns:
        A sorted list of resistance prices (lowest first).
    """
    if len(df) < window * 2:
        return []

    swings = swing_highs(df, window)
    swing_prices = df.loc[swings, "High"].tolist()

    # Filter: only levels above current price
    resistances = [r for r in swing_prices if r > current_price]

    # Remove duplicates (within 0.0001 tolerance)
    unique: list[float] = []
    for r in sorted(resistances):
        if not any(abs(r - u) < 0.0001 for u in unique):
            unique.append(r)

    closest = sorted(unique)[:num_levels]
    logger.debug("Resistance levels found: %s", closest)
    return closest


def resistance_score(
    df: pd.DataFrame,
    current_price: float,
) -> float:
    """Score based on proximity to the nearest resistance level.

    Closer to resistance → more bearish (sell near resistance).
    If price is far from any resistance → neutral.

    Args:
        df: OHLC DataFrame.
        current_price: Current market price.

    Returns:
        A float between 0 and 100.
    """
    resistances = find_resistance_levels(df, current_price, num_levels=1)
    if not resistances:
        return 50.0

    nearest = resistances[0]  # Lowest resistance above price
    distance = nearest - current_price

    from indicators.atr import atr14

    atr_val = atr14(df).iloc[-1]
    if pd.isna(atr_val) or atr_val == 0:
        return 50.0

    ratio = distance / atr_val

    if ratio <= 0.5:
        return 15.0  # Very close to resistance (bearish)
    elif ratio <= 1.0:
        return 30.0
    elif ratio <= 2.0:
        return 45.0
    else:
        return 50.0
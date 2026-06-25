"""Exponential Moving Average (EMA) indicator calculations."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate the Exponential Moving Average for a given period.

    Args:
        series: Price series (typically Close).
        period: EMA window length.

    Returns:
        A pandas Series with the EMA values. The first ``period - 1``
        entries will be ``NaN``.
    """
    return series.ewm(span=period, adjust=False).mean()


def ema20(df: pd.DataFrame) -> pd.Series:
    """EMA with period = 20.

    Args:
        df: OHLC DataFrame with a ``Close`` column.

    Returns:
        EMA-20 series.
    """
    return ema(df["Close"], 20)


def ema50(df: pd.DataFrame) -> pd.Series:
    """EMA with period = 50.

    Args:
        df: OHLC DataFrame with a ``Close`` column.

    Returns:
        EMA-50 series.
    """
    return ema(df["Close"], 50)


def ema200(df: pd.DataFrame) -> pd.Series:
    """EMA with period = 200.

    Args:
        df: OHLC DataFrame with a ``Close`` column.

    Returns:
        EMA-200 series.
    """
    return ema(df["Close"], 200)


def ema_score(df: pd.DataFrame) -> float:
    """Compute a normalised EMA score from 0 to 100.

    Scoring logic:
        - Price above all three EMAs → bullish (high score).
        - Price below all three EMAs → bearish (low score).
        - EMAs stacked in bullish order (20 > 50 > 200) adds points.
        - EMAs stacked in bearish order (20 < 50 < 200) subtracts points.

    Args:
        df: OHLC DataFrame with a ``Close`` column. Must have at least 200 rows.

    Returns:
        A float between 0 and 100. 50 = neutral.
    """
    if len(df) < 200:
        return 50.0

    close = df["Close"].iloc[-1]
    e20 = ema20(df).iloc[-1]
    e50 = ema50(df).iloc[-1]
    e200 = ema200(df).iloc[-1]

    score = 50.0

    # Price position relative to EMAs
    if close > e20:
        score += 5
    elif close < e20:
        score -= 5

    if close > e50:
        score += 5
    elif close < e50:
        score -= 5

    if close > e200:
        score += 5
    elif close < e200:
        score -= 5

    # EMA stack alignment
    if e20 > e50 > e200:
        score += 15
    elif e20 < e50 < e200:
        score -= 15
    elif e20 > e50:
        score += 5
    elif e20 < e50:
        score -= 5

    return max(0.0, min(100.0, score))


def compute_all_ema(
    df: pd.DataFrame,
) -> dict[str, Optional[pd.Series]]:
    """Return a dict with all three EMA series.

    Args:
        df: OHLC DataFrame.

    Returns:
        ``{'ema20': Series, 'ema50': Series, 'ema200': Series}``
    """
    return {
        "ema20": ema20(df) if len(df) >= 20 else None,
        "ema50": ema50(df) if len(df) >= 50 else None,
        "ema200": ema200(df) if len(df) >= 200 else None,
    }
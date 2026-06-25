"""MACD (Moving Average Convergence Divergence) indicator."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def macd_line(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """Calculate the MACD line.

    Args:
        df: OHLC DataFrame with a ``Close`` column.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).

    Returns:
        MACD line series.
    """
    fast_ema = df["Close"].ewm(span=fast, adjust=False).mean()
    slow_ema = df["Close"].ewm(span=slow, adjust=False).mean()
    return fast_ema - slow_ema


def signal_line(macd_series: pd.Series, period: int = 9) -> pd.Series:
    """Calculate the MACD signal line.

    Args:
        macd_series: The MACD line series.
        period: Signal EMA period (default 9).

    Returns:
        Signal line series.
    """
    return macd_series.ewm(span=period, adjust=False).mean()


def histogram(macd_series: pd.Series, signal_series: pd.Series) -> pd.Series:
    """Calculate the MACD histogram.

    Args:
        macd_series: MACD line.
        signal_series: Signal line.

    Returns:
        Histogram series (MACD - Signal).
    """
    return macd_series - signal_series


def macd_score(df: pd.DataFrame) -> float:
    """Compute a normalised MACD score from 0 to 100.

    Scoring logic:
        - MACD line above signal → bullish.
        - Histogram increasing → additional bullish momentum.
        - MACD line above zero → strong bull.

    Args:
        df: OHLC DataFrame. Should have at least 50 rows.

    Returns:
        A float between 0 and 100.
    """
    if len(df) < 35:
        return 50.0

    ml = macd_line(df)
    sl = signal_line(ml)
    hist = histogram(ml, sl)

    current_macd = ml.iloc[-1]
    current_signal = sl.iloc[-1]
    current_hist = hist.iloc[-1]
    prev_hist = hist.iloc[-2] if len(hist) >= 2 else 0.0

    score = 50.0

    # MACD vs signal
    if current_macd > current_signal:
        score += 10
    elif current_macd < current_signal:
        score -= 10

    # MACD vs zero
    if current_macd > 0:
        score += 5
    elif current_macd < 0:
        score -= 5

    # Histogram direction
    if current_hist > prev_hist and current_hist > 0:
        score += 5
    elif current_hist < prev_hist and current_hist < 0:
        score -= 5

    # Crossover detection
    if len(ml) >= 3:
        if ml.iloc[-2] < sl.iloc[-2] and current_macd > current_signal:
            score += 10  # Bullish crossover
        elif ml.iloc[-2] > sl.iloc[-2] and current_macd < current_signal:
            score -= 10  # Bearish crossover

    return max(0.0, min(100.0, score))


def compute_all_macd(
    df: pd.DataFrame,
) -> dict[str, Optional[pd.Series]]:
    """Return a dict with MACD, signal, and histogram series.

    Args:
        df: OHLC DataFrame.

    Returns:
        ``{'macd': Series, 'signal': Series, 'histogram': Series}``
    """
    if len(df) < 35:
        return {"macd": None, "signal": None, "histogram": None}

    ml = macd_line(df)
    sl = signal_line(ml)
    hist = histogram(ml, sl)
    return {"macd": ml, "signal": sl, "histogram": hist}
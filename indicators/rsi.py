"""RSI (Relative Strength Index) indicator."""

from __future__ import annotations

import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index.

    Uses the Wilder smoothing method (exponential moving average of
    gains and losses).

    Args:
        series: Price series (typically Close).
        period: Look-back period (default 14).

    Returns:
        RSI series with values between 0 and 100.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("inf"))
    return 100.0 - (100.0 / (1.0 + rs))


def rsi14(df: pd.DataFrame) -> pd.Series:
    """RSI with the standard 14-period look-back.

    Args:
        df: OHLC DataFrame with a ``Close`` column.

    Returns:
        RSI-14 series.
    """
    return rsi(df["Close"], 14)


def rsi_score(df: pd.DataFrame) -> float:
    """Compute a normalised RSI score from 0 to 100.

    Scoring logic:
        - RSI > 70 → overbought → bearish bias (lower score).
        - RSI < 30 → oversold → bullish bias (higher score).
        - RSI between 40-60 → neutral.

    Args:
        df: OHLC DataFrame. Needs at least 15 rows.

    Returns:
        A float between 0 and 100.
    """
    if len(df) < 15:
        return 50.0

    current_rsi = rsi14(df).iloc[-1]

    if current_rsi >= 70:
        # Overbought — bearish
        return max(0.0, 100.0 - (current_rsi - 70) * 3.33)
    elif current_rsi <= 30:
        # Oversold — bullish
        return min(100.0, (30 - current_rsi) * 3.33)
    elif current_rsi >= 50:
        # Mild bullish
        return 50.0 + (current_rsi - 50) * 1.0
    else:
        # Mild bearish
        return 50.0 + (current_rsi - 50) * 1.0


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute RSI for an arbitrary period.

    Args:
        df: OHLC DataFrame.
        period: Look-back period.

    Returns:
        RSI series.
    """
    return rsi(df["Close"], period)
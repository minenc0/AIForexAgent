"""ATR (Average True Range) volatility indicator."""

from __future__ import annotations

import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate the True Range component for ATR.

    True Range = max(
        High - Low,
        abs(High - previous Close),
        abs(Low - previous Close)
    )

    Args:
        df: OHLC DataFrame with ``High``, ``Low``, ``Close`` columns.

    Returns:
        True Range series (first value is ``NaN``).
    """
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Average True Range.

    Uses the Wilder smoothing method (same as RSI).

    Args:
        df: OHLC DataFrame with ``High``, ``Low``, ``Close`` columns.
        period: ATR look-back period (default 14).

    Returns:
        ATR series.
    """
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def atr14(df: pd.DataFrame) -> pd.Series:
    """ATR with the standard 14-period look-back.

    Args:
        df: OHLC DataFrame.

    Returns:
        ATR-14 series.
    """
    return atr(df, 14)


def atr_score(df: pd.DataFrame) -> float:
    """Compute an ATR-based volatility score.

    This score measures whether the current volatility is favourable
    for trading. Very low ATR → no momentum (neutral). Very high
    ATR → risky (slight penalty). Moderate ATR → good trading
    conditions.

    Args:
        df: OHLC DataFrame. Needs at least 15 rows.

    Returns:
        A float between 0 and 100.
    """
    if len(df) < 15:
        return 50.0

    current_atr = atr14(df).iloc[-1]
    atr_series = atr14(df).dropna()

    if atr_series.empty:
        return 50.0

    avg_atr = atr_series.mean()
    if avg_atr == 0:
        return 50.0

    ratio = current_atr / avg_atr

    # Ratio ~1.0 is normal, give high score
    # Too low (< 0.5) = no volatility, too high (> 2.0) = risky
    if ratio < 0.5:
        return 30.0
    elif ratio < 1.0:
        return 50.0 + (ratio - 0.5) * 60  # 50-80
    elif ratio <= 1.5:
        return 80.0 + (ratio - 1.0) * 20  # 80-90
    elif ratio <= 2.0:
        return 90.0 - (ratio - 1.5) * 40  # 90-70
    else:
        return max(20.0, 70.0 - (ratio - 2.0) * 20)
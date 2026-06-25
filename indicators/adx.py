"""ADX (Average Directional Index) trend strength indicator."""

from __future__ import annotations

import pandas as pd


def _directional_movement(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate +DM and -DM (directional movement).

    Args:
        df: OHLC DataFrame.

    Returns:
        Tuple of (+DM, -DM) series.
    """
    high_diff = df["High"].diff()
    low_diff = -df["Low"].diff()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm = plus_dm.where(
        ~(high_diff > low_diff) | ~(high_diff > 0),
        high_diff,
    )
    minus_dm = minus_dm.where(
        ~(low_diff > high_diff) | ~(low_diff > 0),
        low_diff,
    )

    # Fix: compute properly
    for i in range(1, len(df)):
        up = high_diff.iloc[i]
        down = low_diff.iloc[i]
        if up > down and up > 0:
            plus_dm.iloc[i] = up
            minus_dm.iloc[i] = 0.0
        elif down > up and down > 0:
            plus_dm.iloc[i] = 0.0
            minus_dm.iloc[i] = down
        else:
            plus_dm.iloc[i] = 0.0
            minus_dm.iloc[i] = 0.0

    return plus_dm, minus_dm


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate the ADX, +DI, and -DI.

    Args:
        df: OHLC DataFrame.
        period: Look-back period (default 14).

    Returns:
        DataFrame with columns ``adx``, ``plus_di``, ``minus_di``.
    """
    from indicators.atr import true_range

    tr = true_range(df)
    plus_dm, minus_dm = _directional_movement(df)

    # Wilder smoothing
    atr_smooth = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di_smooth = plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    minus_di_smooth = minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * (plus_di_smooth / atr_smooth.replace(0, float("inf")))
    minus_di = 100.0 * (minus_di_smooth / atr_smooth.replace(0, float("inf")))

    dx_denominator = (plus_di + minus_di).replace(0, float("inf"))
    dx = 100.0 * ((plus_di - minus_di).abs() / dx_denominator)

    adx_line = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame(
        {"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di},
        index=df.index,
    )


def adx14(df: pd.DataFrame) -> pd.DataFrame:
    """ADX with the standard 14-period look-back.

    Args:
        df: OHLC DataFrame.

    Returns:
        DataFrame with ``adx``, ``plus_di``, ``minus_di`` columns.
    """
    return adx(df, 14)


def adx_score(df: pd.DataFrame) -> float:
    """Compute a normalised ADX score from 0 to 100.

    Scoring logic:
        - ADX > 25 → strong trend.
        - +DI > -DI → bullish trend.
        - -DI > +DI → bearish trend.

    Args:
        df: OHLC DataFrame. Needs at least 30 rows.

    Returns:
        A float between 0 and 100.
    """
    if len(df) < 30:
        return 50.0

    adx_data = adx14(df)
    current_adx = adx_data["adx"].iloc[-1]
    current_plus_di = adx_data["plus_di"].iloc[-1]
    current_minus_di = adx_data["minus_di"].iloc[-1]

    if pd.isna(current_adx):
        return 50.0

    score = 50.0

    # Trend strength
    if current_adx > 50:
        score += 5
    elif current_adx > 25:
        score += 3
    else:
        score -= 5  # Weak trend

    # Directional
    if current_plus_di > current_minus_di:
        directional = (current_plus_di - current_minus_di) / max(current_plus_di, 0.01)
        score += min(15, directional * 15)
    else:
        directional = (current_minus_di - current_plus_di) / max(current_minus_di, 0.01)
        score -= min(15, directional * 15)

    return max(0.0, min(100.0, score))
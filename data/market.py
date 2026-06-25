"""Market data fetching via yfinance with caching and retry."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Optional

import pandas as pd
import yfinance as yf

from config import Timeframe
from utils.logger import logger

# Cache TTL per timeframe (seconds)
_CACHE_TTL: dict[Timeframe, int] = {
    Timeframe.M5: 30,
    Timeframe.M15: 60,
    Timeframe.M30: 120,
    Timeframe.H1: 300,
    Timeframe.H4: 600,
}

# In-memory cache: (pair, timeframe) → (timestamp, DataFrame)
_market_cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}


def _yf_interval(tf: Timeframe) -> str:
    """Map our Timeframe enum to yfinance interval string."""
    return tf.value


def _yf_period(tf: Timeframe) -> str:
    """Return an appropriate yfinance period string for the timeframe.

    Ensures enough bars for all indicator calculations (at least 250 bars).
    """
    mapping: dict[Timeframe, str] = {
        Timeframe.M5: "5d",
        Timeframe.M15: "10d",
        Timeframe.M30: "20d",
        Timeframe.H1: "30d",
        Timeframe.H4: "60d",
    }
    return mapping.get(tf, "30d")


def _fetch_with_retry(
    ticker: str,
    interval: str,
    period: str,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from yfinance with retry logic.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. ``'EURUSD=X'``).
        interval: Bar interval (e.g. ``'1h'``).
        period: Look-back period (e.g. ``'30d'``).
        max_retries: Maximum number of retry attempts.
        backoff: Exponential backoff multiplier in seconds.

    Returns:
        A DataFrame with columns ``Open, High, Low, Close, Volume``
        or ``None`` on failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(
                ticker,
                interval=interval,
                period=period,
                progress=False,
                auto_adjust=True,
            )
            if data.empty:
                logger.warning(
                    "yfinance returned empty data for %s (attempt %d/%d)",
                    ticker, attempt, max_retries,
                )
                continue

            # Ensure standard column names
            data.columns = [c.strip() for c in data.columns]
            if "Close" not in data.columns:
                # Multi-level columns
                data.columns = data.columns.get_level_values(0)

            # Keep only needed columns
            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col not in data.columns:
                    data[col] = 0.0

            data = data[["Open", "High", "Low", "Close", "Volume"]]
            data = data.dropna(subset=["Close"])
            data = data[data["Close"] > 0]

            if data.empty:
                continue

            logger.info(
                "Fetched %d bars for %s @ %s",
                len(data), ticker, interval,
            )
            return data

        except Exception as e:
            logger.error(
                "yfinance error for %s (attempt %d/%d): %s",
                ticker, attempt, max_retries, e,
            )
            if attempt < max_retries:
                time.sleep(backoff ** attempt)

    logger.error("All retries exhausted for %s @ %s", ticker, interval)
    return None


def _pair_to_ticker(pair: str) -> str:
    """Convert a forex pair string (e.g. ``'EURUSD'``) to a yfinance ticker.

    Args:
        pair: Currency pair without separator (e.g. ``'EURUSD'``).

    Returns:
        Yahoo Finance ticker (e.g. ``'EURUSD=X'``).
    """
    return f"{pair}=X"


def get_market_data(
    pair: str,
    timeframe: Timeframe,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data for a forex pair at a given timeframe.

    Uses an in-memory time-based cache to avoid redundant API calls.

    Args:
        pair: Forex pair (e.g. ``'EURUSD'``).
        timeframe: Bar period.
        use_cache: Whether to use the cache.

    Returns:
        OHLCV DataFrame. Raises ``ValueError`` if data cannot be fetched.
    """
    key = (pair, timeframe.value)

    if use_cache and key in _market_cache:
        ts, data = _market_cache[key]
        ttl = _CACHE_TTL.get(timeframe, 300)
        if time.time() - ts < ttl:
            logger.debug("Cache hit for %s @ %s", pair, timeframe.value)
            return data.copy()

    ticker = _pair_to_ticker(pair)
    interval = _yf_interval(timeframe)
    period = _yf_period(timeframe)

    data = _fetch_with_retry(ticker, interval, period)

    if data is None or data.empty:
        raise ValueError(f"Cannot fetch market data for {pair} @ {timeframe.value}")

    _market_cache[key] = (time.time(), data)
    return data.copy()


def get_current_price(pair: str) -> float:
    """Get the latest available close price for a pair.

    Args:
        pair: Forex pair string.

    Returns:
        The most recent close price.

    Raises:
        ValueError: If no price data is available.
    """
    df = get_market_data(pair, Timeframe.M5)
    if df.empty:
        raise ValueError(f"No price data available for {pair}")
    return float(df["Close"].iloc[-1])
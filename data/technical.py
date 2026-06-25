"""Aggregate technical analysis — computes all indicators for a given pair/tf."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from config import Timeframe
from data.market import get_market_data
from indicators import (
    atr,
    candlestick,
    ema,
    macd,
    resistance,
    rsi,
    support,
    adx as adx_mod,
)
from utils.logger import logger


@dataclass
class TechnicalResult:
    """Container for all computed technical indicator values and scores."""

    pair: str = ""
    timeframe: str = ""
    current_price: float = 0.0
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    rsi14: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    atr14: Optional[float] = None
    adx14: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)
    candlestick_patterns: list[str] = field(default_factory=list)
    ema_score: float = 50.0
    macd_score: float = 50.0
    rsi_score: float = 50.0
    atr_score: float = 50.0
    adx_score: float = 50.0
    support_score: float = 50.0
    resistance_score: float = 50.0
    candlestick_score: float = 50.0
    df: Optional[pd.DataFrame] = None
    ema20_series: Optional[pd.Series] = None
    ema50_series: Optional[pd.Series] = None
    ema200_series: Optional[pd.Series] = None


def compute_technical_analysis(
    pair: str,
    timeframe: Timeframe,
) -> TechnicalResult:
    """Run all technical indicators on a pair/timeframe.

    Args:
        pair: Forex pair (e.g. ``'EURUSD'``).
        timeframe: Bar period.

    Returns:
        A ``TechnicalResult`` with all values populated.
    """
    logger.info("Computing technical analysis for %s @ %s", pair, timeframe.value)
    result = TechnicalResult(pair=pair, timeframe=timeframe.value)

    df = get_market_data(pair, timeframe)
    result.df = df
    result.current_price = float(df["Close"].iloc[-1])

    # EMA
    all_ema = ema.compute_all_ema(df)
    result.ema20_series = all_ema["ema20"]
    result.ema50_series = all_ema["ema50"]
    result.ema200_series = all_ema["ema200"]

    if all_ema["ema20"] is not None and not all_ema["ema20"].isna().iloc[-1]:
        result.ema20 = float(all_ema["ema20"].iloc[-1])
    if all_ema["ema50"] is not None and not all_ema["ema50"].isna().iloc[-1]:
        result.ema50 = float(all_ema["ema50"].iloc[-1])
    if all_ema["ema200"] is not None and not all_ema["ema200"].isna().iloc[-1]:
        result.ema200 = float(all_ema["ema200"].iloc[-1])
    result.ema_score = ema.ema_score(df)

    # RSI
    rsi_series = rsi.rsi14(df)
    if not rsi_series.isna().iloc[-1]:
        result.rsi14 = float(rsi_series.iloc[-1])
    result.rsi_score = rsi.rsi_score(df)

    # MACD
    all_macd = macd.compute_all_macd(df)
    if all_macd["macd"] is not None and not all_macd["macd"].isna().iloc[-1]:
        result.macd_line = float(all_macd["macd"].iloc[-1])
        result.macd_signal = float(all_macd["signal"].iloc[-1])
        result.macd_hist = float(all_macd["histogram"].iloc[-1])
    result.macd_score = macd.macd_score(df)

    # ATR
    atr_series = atr.atr14(df)
    if not atr_series.isna().iloc[-1]:
        result.atr14 = float(atr_series.iloc[-1])
    result.atr_score = atr.atr_score(df)

    # ADX
    adx_data = adx_mod.adx14(df)
    if not adx_data["adx"].isna().iloc[-1]:
        result.adx14 = float(adx_data["adx"].iloc[-1])
        result.plus_di = float(adx_data["plus_di"].iloc[-1])
        result.minus_di = float(adx_data["minus_di"].iloc[-1])
    result.adx_score = adx_mod.adx_score(df)

    # Support / Resistance
    result.support_levels = support.find_support_levels(df, result.current_price)
    result.resistance_levels = resistance.find_resistance_levels(df, result.current_price)
    result.support_score = support.support_score(df, result.current_price)
    result.resistance_score = resistance.resistance_score(df, result.current_price)

    # Candlestick patterns
    result.candlestick_patterns = candlestick.get_detected_patterns(df)
    result.candlestick_score = candlestick.candlestick_score(df)

    logger.info(
        "Technical analysis complete for %s @ %s | EMA=%.1f RSI=%.1f MACD=%.1f ATR=%.5f ADX=%.1f",
        pair, timeframe.value, result.ema_score, result.rsi_score,
        result.macd_score, result.atr14 or 0, result.adx14 or 0,
    )

    return result


def compute_weighted_technical_score(
    result: TechnicalResult,
    weights: dict[str, float],
) -> float:
    """Calculate the weighted technical score.

    Args:
        result: Populated ``TechnicalResult``.
        weights: Mapping of indicator name to weight (must sum to the
            technical portion, typically 80 out of 100).

    Returns:
        A weighted score between 0 and 100.
    """
    w_ema = weights.get("ema", 15)
    w_macd = weights.get("macd", 10)
    w_rsi = weights.get("rsi", 10)
    w_atr = weights.get("atr", 5)
    w_adx = weights.get("adx", 10)
    w_trend = weights.get("trend", 15)
    w_sr = weights.get("support_resistance", 10)
    w_candle = weights.get("candlestick", 10)

    sr_score = (result.support_score + result.resistance_score) / 2.0
    # Use EMA score as a proxy for trend score since trend is EMA-based
    trend_score = result.ema_score

    total_weight = w_ema + w_macd + w_rsi + w_atr + w_adx + w_trend + w_sr + w_candle
    if total_weight == 0:
        return 50.0

    weighted = (
        result.ema_score * w_ema
        + result.macd_score * w_macd
        + result.rsi_score * w_rsi
        + result.atr_score * w_atr
        + result.adx_score * w_adx
        + trend_score * w_trend
        + sr_score * w_sr
        + result.candlestick_score * w_candle
    ) / total_weight

    return max(0.0, min(100.0, weighted))
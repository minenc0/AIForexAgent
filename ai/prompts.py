"""Prompt engineering — builds structured prompts for AI reasoning."""

from __future__ import annotations

from typing import Any, Optional

from utils.logger import logger


SYSTEM_PROMPT = """You are an expert Forex Trading Analyst AI. Your role is to analyze
technical data, correlation signals, news impact, and risk parameters to provide
a reasoned trading decision.

You must:
1. Evaluate ALL data points objectively
2. Consider both bullish and bearish scenarios
3. Provide a clear, definitive decision: BUY, SELL, or NO TRADE
4. Explain your reasoning step by step
5. Include a confidence level (0-100%)
6. Never hallucinate data — only use what is provided

You are NOT a chatbot. You are a decision-making agent.
Think carefully and provide your analysis in a structured format.

Your response MUST follow this exact format:

DECISION: [BUY/SELL/NO TRADE]
CONFIDENCE: [0-100]
REASONING: [Your detailed reasoning, 3-5 sentences explaining why]

Do NOT add any other text outside this format."""


def build_analysis_prompt(
    pair: str,
    timeframe: str,
    current_price: float,
    trend: str,
    ema_data: dict[str, Any],
    rsi_data: dict[str, Any],
    macd_data: dict[str, Any],
    atr_data: dict[str, Any],
    adx_data: dict[str, Any],
    support_data: dict[str, Any],
    resistance_data: dict[str, Any],
    candlestick_data: dict[str, Any],
    correlation_data: dict[str, Any],
    news_data: dict[str, Any],
    session: str,
    risk_data: dict[str, Any],
    technical_score: float,
    correlation_score: float,
    multi_timeframe_data: Optional[dict[str, Any]] = None,
) -> str:
    """Build a comprehensive analysis prompt for the LLM.

    Instead of asking a simple BUY/SELL question, this function constructs
    a detailed prompt containing ALL gathered data so the LLM can perform
    proper reasoning.

    Args:
        pair: Forex pair (e.g. ``'EURUSD'``).
        timeframe: Current timeframe string.
        current_price: Latest close price.
        trend: Detected trend direction.
        ema_data: Dictionary with EMA values and score.
        rsi_data: Dictionary with RSI value and score.
        macd_data: Dictionary with MACD values and score.
        atr_data: Dictionary with ATR value and score.
        adx_data: Dictionary with ADX values and score.
        support_data: Dictionary with support levels.
        resistance_data: Dictionary with resistance levels.
        candlestick_data: Dictionary with detected patterns.
        correlation_data: Dictionary with correlation analysis results.
        news_data: Dictionary with news filter results.
        session: Active trading session.
        risk_data: Dictionary with risk management parameters.
        technical_score: Aggregated technical score (0-100).
        correlation_score: Correlation analysis score (0-100).
        multi_timeframe_data: Optional multi-timeframe alignment data.

    Returns:
        A formatted prompt string ready for the LLM.
    """
    prompt = f"""FOREX ANALYSIS REQUEST
====================

PAIR: {pair}
TIMEFRAME: {timeframe}
CURRENT PRICE: {current_price:.5f}
ACTIVE SESSION: {session}

--- TREND ---
Direction: {trend}

--- EMA (Score: {ema_data.get('score', 'N/A')}) ---
EMA 20: {ema_data.get('ema20', 'N/A')}
EMA 50: {ema_data.get('ema50', 'N/A')}
EMA 200: {ema_data.get('ema200', 'N/A')}

--- RSI (Score: {rsi_data.get('score', 'N/A')}) ---
RSI 14: {rsi_data.get('rsi14', 'N/A')}

--- MACD (Score: {macd_data.get('score', 'N/A')}) ---
MACD Line: {macd_data.get('macd_line', 'N/A')}
Signal Line: {macd_data.get('signal', 'N/A')}
Histogram: {macd_data.get('histogram', 'N/A')}

--- ATR (Score: {atr_data.get('score', 'N/A')}) ---
ATR 14: {atr_data.get('atr14', 'N/A')}

--- ADX (Score: {adx_data.get('score', 'N/A')}) ---
ADX: {adx_data.get('adx14', 'N/A')}
+DI: {adx_data.get('plus_di', 'N/A')}
-DI: {adx_data.get('minus_di', 'N/A')}

--- SUPPORT & RESISTANCE ---
Support Levels: {support_data.get('levels', [])}
Resistance Levels: {resistance_data.get('levels', [])}

--- CANDLESTICK PATTERNS ---
Detected: {candlestick_data.get('patterns', ['None'])}
Score: {candlestick_data.get('score', 'N/A')}

--- CORRELATION ANALYSIS (Score: {correlation_score}) ---
{correlation_data.get('details', 'No correlation data')}

--- NEWS ---
Status: {news_data.get('status', 'N/A')}
Upcoming Events: {news_data.get('event_count', 0)}

--- RISK MANAGEMENT ---
Entry: {risk_data.get('entry', 'N/A')}
Stop Loss: {risk_data.get('stop_loss', 'N/A')}
Take Profit: {risk_data.get('take_profit', 'N/A')}
Risk:Reward: 1:{risk_data.get('risk_reward', 'N/A')}

--- AGGREGATE SCORES ---
Technical Score: {technical_score:.1f}/100
Correlation Score: {correlation_score:.1f}/100
Combined Score: {(technical_score * 0.8 + correlation_score * 0.2):.1f}/100"""

    if multi_timeframe_data:
        prompt += f"""

--- MULTI TIMEFRAME ANALYSIS ---
Alignment: {multi_timeframe_data.get('alignment', 'N/A')}
Details:
"""
        for tf_name, tf_info in multi_timeframe_data.get("timeframes", {}).items():
            prompt += f"  {tf_name}: {tf_info}\n"

    prompt += """

Based on ALL the data above, provide your analysis and decision.
Remember: analyze everything before deciding. Do not rush to a conclusion.
Consider the correlation confirmation, news risk, and multi-timeframe alignment."""

    logger.debug("Analysis prompt built for %s @ %s (%d chars)", pair, timeframe, len(prompt))
    return prompt


def parse_ai_response(response_text: str) -> dict[str, Any]:
    """Parse the structured AI response into a dictionary.

    Expects the format:
        DECISION: BUY
        CONFIDENCE: 85
        REASONING: Detailed explanation...

    Args:
        response_text: Raw LLM response text.

    Returns:
        Dictionary with keys ``decision``, ``confidence``, ``reasoning``.
    """
    result = {
        "decision": "NO TRADE",
        "confidence": 50,
        "reasoning": response_text,
    }

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line.upper().startswith("DECISION:"):
            decision = line.split(":", 1)[1].strip().upper()
            if decision in ("BUY", "SELL", "NO TRADE", "STRONG BUY", "STRONG SELL"):
                result["decision"] = decision
            else:
                result["decision"] = "NO TRADE"

        elif line.upper().startswith("CONFIDENCE:"):
            try:
                conf_str = line.split(":", 1)[1].strip().replace("%", "")
                result["confidence"] = max(0, min(100, int(conf_str)))
            except (ValueError, IndexError):
                result["confidence"] = 50

        elif line.upper().startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()

    return result
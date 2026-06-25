"""Prompt Agent — assembles the final prompt and calls the LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ai.openai_client import chat_completion
from ai.prompts import SYSTEM_PROMPT, build_analysis_prompt, parse_ai_response
from config import AIModel
from utils.logger import logger


@dataclass
class PromptAgentResult:
    """Result of the AI reasoning via LLM."""

    decision: str  # BUY, SELL, NO TRADE, Strong Buy, Strong Sell
    confidence: int  # 0-100
    reasoning: str
    model_used: str
    raw_response: str


class PromptAgent:
    """Assembles all analysis data into a structured prompt and sends it
    to the LLM for reasoning.

    This agent does NOT make the final decision itself. It collects all
    data, builds a comprehensive prompt, sends it to GPT, and parses
    the response.
    """

    def __init__(self, model: AIModel = AIModel.GPT4O_MINI) -> None:
        """Initialise the PromptAgent.

        Args:
            model: The OpenAI model to use for reasoning.
        """
        self.model = model.value

    def build_and_send(
        self,
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
    ) -> PromptAgentResult:
        """Build the prompt, send to LLM, and parse the response.

        Args:
            pair: Forex pair.
            timeframe: Timeframe string.
            current_price: Current market price.
            trend: Trend direction.
            ema_data: EMA analysis data dict.
            rsi_data: RSI analysis data dict.
            macd_data: MACD analysis data dict.
            atr_data: ATR analysis data dict.
            adx_data: ADX analysis data dict.
            support_data: Support levels dict.
            resistance_data: Resistance levels dict.
            candlestick_data: Candlestick pattern data dict.
            correlation_data: Correlation analysis dict.
            news_data: News filter dict.
            session: Active session.
            risk_data: Risk parameters dict.
            technical_score: Aggregated technical score.
            correlation_score: Correlation score.
            multi_timeframe_data: Multi-timeframe alignment data.

        Returns:
            A ``PromptAgentResult`` with the LLM's decision.
        """
        logger.info(
            "PromptAgent building prompt for %s @ %s (model=%s)",
            pair, timeframe, self.model,
        )

        # Build the comprehensive prompt
        user_prompt = build_analysis_prompt(
            pair=pair,
            timeframe=timeframe,
            current_price=current_price,
            trend=trend,
            ema_data=ema_data,
            rsi_data=rsi_data,
            macd_data=macd_data,
            atr_data=atr_data,
            adx_data=adx_data,
            support_data=support_data,
            resistance_data=resistance_data,
            candlestick_data=candlestick_data,
            correlation_data=correlation_data,
            news_data=news_data,
            session=session,
            risk_data=risk_data,
            technical_score=technical_score,
            correlation_score=correlation_score,
            multi_timeframe_data=multi_timeframe_data,
        )

        # Call the LLM
        try:
            raw_response = chat_completion(
                model=self.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=1024,
            )
        except RuntimeError as e:
            logger.error("PromptAgent: LLM call failed: %s", e)
            return PromptAgentResult(
                decision="NO TRADE",
                confidence=50,
                reasoning=f"AI reasoning unavailable: {e}",
                model_used=self.model,
                raw_response="",
            )

        # Parse the response
        parsed = parse_ai_response(raw_response)

        logger.info(
            "PromptAgent result: %s (confidence=%d) using %s",
            parsed["decision"],
            parsed["confidence"],
            self.model,
        )

        return PromptAgentResult(
            decision=parsed["decision"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            model_used=self.model,
            raw_response=raw_response,
        )
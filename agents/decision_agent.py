"""Decision Agent — orchestrates all sub-agents and produces the final decision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from config import (
    AIModel,
    Decision,
    IndicatorWeights,
    RiskConfig,
    Timeframe,
    score_to_decision,
)
from agents.analysis_agent import TechnicalAgent, TechnicalAgentResult
from agents.correlation_agent import CorrelationAgent, CorrelationAgentResult
from agents.memory_agent import MemoryAgent, MemoryResult
from agents.news_agent import NewsAgent, NewsResult
from agents.prompt_agent import PromptAgent, PromptAgentResult
from agents.risk_agent import RiskAgent, RiskAgentResult
from agents.trend_agent import TrendAgent, TrendResult
from data.economic_calendar import get_current_session
from utils.database import TradeDatabase
from utils.logger import logger


@dataclass
class FullAnalysisResult:
    """Complete result from the orchestrated analysis pipeline."""

    # Core outputs
    pair: str = ""
    timeframe: str = ""
    decision: str = "NO TRADE"
    confidence: float = 50.0
    entry: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward: float = 2.0

    # Scores
    technical_score: float = 50.0
    correlation_score: float = 50.0
    news_score: float = 100.0
    combined_score: float = 50.0

    # Context
    trend: str = "neutral"
    session: str = ""
    current_price: float = 0.0

    # Detailed results from each agent
    trend_result: Optional[TrendResult] = None
    technical_result: Optional[TechnicalAgentResult] = None
    correlation_result: Optional[CorrelationAgentResult] = None
    news_result: Optional[NewsResult] = None
    risk_result: Optional[RiskAgentResult] = None
    prompt_result: Optional[PromptAgentResult] = None
    memory_result: Optional[MemoryResult] = None

    # AI reasoning
    ai_reasoning: str = ""
    use_ai_reasoning: bool = False

    # Multi-timeframe data
    multi_timeframe: Optional[dict[str, Any]] = None


# NOTE: This file is named decision_agent.py but contains the orchestrator.
# The DecisionAgent class is the central coordinator.

@dataclass
class DecisionResult:
    """Result from the decision agent."""
    decision: str
    confidence: float
    reasoning: str


class DecisionAgent:
    """Orchestrates all sub-agents to produce a final trading decision.

    Pipeline:
        1. TrendAgent → direction
        2. TechnicalAgent → technical score
        3. CorrelationAgent → correlation score
        4. NewsAgent → news filter
        5. RiskAgent → entry/SL/TP
        6. MemoryAgent → adaptive weights
        7. (Optional) PromptAgent → AI reasoning
        8. Combine → final decision
    """

    def __init__(
        self,
        model: AIModel = AIModel.GPT4O_MINI,
        weights: Optional[IndicatorWeights] = None,
        risk_config: Optional[RiskConfig] = None,
        balance: float = 10000.0,
        db: Optional[TradeDatabase] = None,
        use_correlation: bool = True,
        use_news: bool = True,
        use_multi_tf: bool = True,
        use_candlestick: bool = True,
        use_ai_reasoning: bool = True,
        use_atr: bool = True,
        use_sr: bool = True,
        use_ema: bool = True,
        use_macd: bool = True,
        use_rsi: bool = True,
        use_adx: bool = True,
    ) -> None:
        """Initialise the DecisionAgent.

        Args:
            model: AI model for reasoning.
            weights: Indicator weights.
            risk_config: Risk management config.
            balance: Account balance.
            db: Database instance.
            use_correlation: Enable correlation filter.
            use_news: Enable news filter.
            use_multi_tf: Enable multi-timeframe analysis.
            use_candlestick: Enable candlestick pattern detection.
            use_ai_reasoning: Enable LLM reasoning.
            use_atr: Enable ATR indicator.
            use_sr: Enable support/resistance.
            use_ema: Enable EMA indicators.
            use_macd: Enable MACD indicator.
            use_rsi: Enable RSI indicator.
            use_adx: Enable ADX indicator.
        """
        self.model = model
        self.weights = weights or IndicatorWeights()
        self.risk_config = risk_config or RiskConfig()
        self.balance = balance
        self.db = db or TradeDatabase(":memory:")

        # Feature toggles
        self.use_correlation = use_correlation
        self.use_news = use_news
        self.use_multi_tf = use_multi_tf
        self.use_candlestick = use_candlestick
        self.use_ai_reasoning = use_ai_reasoning
        self.use_atr = use_atr
        self.use_sr = use_sr
        self.use_ema = use_ema
        self.use_macd = use_macd
        self.use_rsi = use_rsi
        self.use_adx = use_adx

        # Sub-agents
        self.trend_agent = TrendAgent()
        self.technical_agent = TechnicalAgent(self.weights)
        self.correlation_agent = CorrelationAgent()
        self.news_agent = NewsAgent()
        self.risk_agent = RiskAgent(self.risk_config, self.balance)
        self.memory_agent = MemoryAgent(self.db, self.weights)
        self.prompt_agent = PromptAgent(model)

    def run(
        self,
        pair: str,
        timeframe: Timeframe,
    ) -> FullAnalysisResult:
        """Execute the full analysis pipeline.

        Args:
            pair: Forex pair to analyse.
            timeframe: Primary timeframe.

        Returns:
            A ``FullAnalysisResult`` with all analysis data and the
            final trading decision.
        """
        logger.info("=" * 60)
        logger.info("DecisionAgent: Starting analysis for %s @ %s", pair, timeframe.value)
        logger.info("=" * 60)

        result = FullAnalysisResult(
            pair=pair,
            timeframe=timeframe.value,
            session=get_current_session(),
        )

        # Step 1: Trend Analysis
        trend_result = self.trend_agent.analyse(pair, timeframe, use_higher_tf=self.use_multi_tf)
        result.trend_result = trend_result
        result.trend = trend_result.direction

        # Step 2: Technical Analysis
        tech_result = self.technical_agent.analyse(pair, timeframe)
        result.technical_result = tech_result
        result.technical_score = tech_result.technical_score

        # Get current price
        if tech_result.raw:
            result.current_price = tech_result.raw.current_price

        # Step 3: News Analysis
        if self.use_news:
            news_result = self.news_agent.analyse(pair)
            result.news_result = news_result
            result.news_score = news_result.news_score
        else:
            result.news_score = 100.0

        # Early exit: high-impact news → NO TRADE
        if result.news_score == 0.0:
            result.decision = "NO TRADE"
            result.confidence = 95.0
            result.ai_reasoning = "BLOCKED: High-impact economic event within 30 minutes. Trading is not recommended."
            logger.info("DecisionAgent: NO TRADE due to high-impact news")
            return result

        # Step 4: Correlation Analysis
        main_direction = trend_result.direction
        if self.use_correlation:
            corr_result = self.correlation_agent.analyse(pair, main_direction, timeframe)
            result.correlation_result = corr_result
            result.correlation_score = corr_result.correlation_score
        else:
            result.correlation_score = 50.0

        # Step 5: Multi-Timeframe Analysis
        multi_tf_data: dict[str, Any] = {"alignment": "N/A", "timeframes": {}}
        if self.use_multi_tf:
            tf_hierarchy = [Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5]
            aligned_count = 0
            for tf in tf_hierarchy:
                try:
                    tf_trend = self.trend_agent.analyse(pair, tf, use_higher_tf=False)
                    multi_tf_data["timeframes"][tf.value] = (
                        f"{tf_trend.direction} ({tf_trend.strength}) "
                        f"score={tf_trend.score:.0f}"
                    )
                    if tf_trend.direction == main_direction and main_direction != "neutral":
                        aligned_count += 1
                except Exception:
                    multi_tf_data["timeframes"][tf.value] = "Data unavailable"

            total = len(tf_hierarchy)
            if aligned_count == total:
                multi_tf_data["alignment"] = "Strong Alignment"
            elif aligned_count >= total * 0.6:
                multi_tf_data["alignment"] = "Moderate Alignment"
            else:
                multi_tf_data["alignment"] = "Weak/No Alignment"

        result.multi_timeframe = multi_tf_data

        # Step 6: Memory Agent (adaptive weights)
        memory_result = self.memory_agent.evaluate_and_adapt()
        result.memory_result = memory_result

        # Step 7: Calculate combined score
        w = self.weights
        combined = (
            result.technical_score * (w.ema + w.macd + w.rsi + w.atr + w.adx + w.trend + w.support_resistance + w.candlestick) / 100.0
            + result.correlation_score * (w.correlation / 100.0)
            + result.news_score * (w.news / 100.0)
        )

        # Apply correlation confidence modifier
        if result.correlation_result:
            combined += result.correlation_result.confidence_modifier

        combined = max(0.0, min(100.0, combined))
        result.combined_score = round(combined, 2)

        # Step 8: Determine base decision from score
        decision = score_to_decision(result.combined_score)

        # Step 9: Risk Management
        risk_result = self.risk_agent.analyse(pair, timeframe, decision)
        result.risk_result = risk_result
        result.entry = risk_result.entry
        result.stop_loss = risk_result.stop_loss
        result.take_profit = risk_result.take_profit
        result.risk_reward = risk_result.risk_reward

        # Step 10: AI Reasoning (optional)
        if self.use_ai_reasoning:
            prompt_result = self.prompt_agent.build_and_send(
                pair=pair,
                timeframe=timeframe.value,
                current_price=result.current_price,
                trend=result.trend,
                ema_data={
                    "ema20": tech_result.raw.ema20 if tech_result.raw else None,
                    "ema50": tech_result.raw.ema50 if tech_result.raw else None,
                    "ema200": tech_result.raw.ema200 if tech_result.raw else None,
                    "score": tech_result.individual_scores.get("EMA", 50),
                },
                rsi_data={
                    "rsi14": tech_result.raw.rsi14 if tech_result.raw else None,
                    "score": tech_result.individual_scores.get("RSI", 50),
                },
                macd_data={
                    "macd_line": tech_result.raw.macd_line if tech_result.raw else None,
                    "signal": tech_result.raw.macd_signal if tech_result.raw else None,
                    "histogram": tech_result.raw.macd_hist if tech_result.raw else None,
                    "score": tech_result.individual_scores.get("MACD", 50),
                },
                atr_data={
                    "atr14": tech_result.raw.atr14 if tech_result.raw else None,
                    "score": tech_result.individual_scores.get("ATR", 50),
                },
                adx_data={
                    "adx14": tech_result.raw.adx14 if tech_result.raw else None,
                    "plus_di": tech_result.raw.plus_di if tech_result.raw else None,
                    "minus_di": tech_result.raw.minus_di if tech_result.raw else None,
                    "score": tech_result.individual_scores.get("ADX", 50),
                },
                support_data={
                    "levels": tech_result.raw.support_levels if tech_result.raw else [],
                },
                resistance_data={
                    "levels": tech_result.raw.resistance_levels if tech_result.raw else [],
                },
                candlestick_data={
                    "patterns": tech_result.raw.candlestick_patterns if tech_result.raw else [],
                    "score": tech_result.individual_scores.get("Candlestick", 50),
                },
                correlation_data={
                    "details": result.correlation_result.details if result.correlation_result else "Disabled",
                    "score": result.correlation_score,
                },
                news_data={
                    "status": result.news_result.status if result.news_result else "N/A",
                    "event_count": result.news_result.event_count if result.news_result else 0,
                },
                session=result.session,
                risk_data={
                    "entry": risk_result.entry,
                    "stop_loss": risk_result.stop_loss,
                    "take_profit": risk_result.take_profit,
                    "risk_reward": risk_result.risk_reward,
                },
                technical_score=result.technical_score,
                correlation_score=result.correlation_score,
                multi_timeframe_data=multi_tf_data,
            )
            result.prompt_result = prompt_result

            # Use AI decision if confidence is reasonable
            ai_confidence = prompt_result.confidence
            if ai_confidence >= 60:
                result.decision = prompt_result.decision
                result.confidence = float(ai_confidence)
                result.ai_reasoning = prompt_result.reasoning
            else:
                result.decision = decision.value
                result.confidence = result.combined_score
                result.ai_reasoning = (
                    f"AI confidence too low ({ai_confidence}%). "
                    f"Falling back to score-based decision: {decision.value}"
                )
        else:
            result.decision = decision.value
            result.confidence = result.combined_score
            result.ai_reasoning = f"Score-based decision (AI reasoning disabled). Combined score: {result.combined_score:.1f}"

        # Step 11: Save to database
        self.db.save_trade({
            "pair": pair,
            "timeframe": timeframe.value,
            "decision": result.decision,
            "confidence": result.confidence,
            "entry": result.entry,
            "sl": result.stop_loss,
            "tp": result.take_profit,
            "risk_reward": result.risk_reward,
            "technical_score": result.technical_score,
            "correlation_score": result.correlation_score,
            "trend": result.trend,
            "reason": result.ai_reasoning,
            "profit_loss": 0.0,
        })

        # Evaluate pending trades
        if result.current_price > 0:
            self.db.evaluate_pending_trades(result.current_price, pair)

        logger.info(
            "DecisionAgent FINAL: %s | %s | Confidence=%.1f | Score=%.1f | %s",
            pair, result.decision, result.confidence, result.combined_score,
            result.ai_reasoning[:100],
        )

        return result
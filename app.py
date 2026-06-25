"""AI Forex Decision Agent — Streamlit Dashboard.

A professional multi-agent AI trading system that analyses forex pairs
using technical indicators, correlation analysis, news filtering, and
LLM-based reasoning to produce BUY/SELL/NO TRADE decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from streamlit import columns, expander, metric, spinner, tabs

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import ALL_PAIRS, AIModel, Timeframe, score_to_decision
from agents.decision_agent import DecisionAgent, FullAnalysisResult
from utils.backtest import BacktestResult, compute_backtest
from utils.charts import (
    create_candlestick_chart,
    create_correlation_bar,
    create_score_gauge,
)
from utils.config_loader import (
    build_app_config,
    get_db_path,
)
from utils.database import TradeDatabase
from utils.logger import logger

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Forex Decision Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
<style>
    /* Global overrides */
    .stApp {
        background-color: #0E1117;
    }
    section[data-testid="stSidebar"] {
        background-color: #1A1D23;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA !important;
    }
    p, span, div, label {
        color: #B0BEC5 !important;
    }
    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 1.6rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #78909C !important;
        font-size: 0.85rem !important;
    }
    /* Sidebar selectbox / checkbox */
    .stSelectbox label, .stCheckbox label {
        color: #CFD8DC !important;
    }
    /* Decision banner */
    .decision-banner {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin: 0.5rem 0;
    }
    .decision-buy {
        background: linear-gradient(135deg, #00C853, #00E676);
        color: #0E1117;
    }
    .decision-sell {
        background: linear-gradient(135deg, #FF1744, #FF5252);
        color: #FFFFFF;
    }
    .decision-no-trade {
        background: linear-gradient(135deg, #FF9800, #FFB74D);
        color: #0E1117;
    }
    .decision-wait {
        background: linear-gradient(135deg, #2196F3, #64B5F6);
        color: #0E1117;
    }
    .decision-strong-buy {
        background: linear-gradient(135deg, #00C853, #69F0AE);
        color: #0E1117;
    }
    .decision-strong-sell {
        background: linear-gradient(135deg, #D50000, #FF1744);
        color: #FFFFFF;
    }
    /* Score cards */
    .score-card {
        background: #1A1D23;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
    }
    /* Reasoning box */
    .reasoning-box {
        background: #1A1D23;
        border-left: 4px solid #1DB954;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    /* Backtest table */
    .dataframe {
        font-size: 0.85rem !important;
    }
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_decision_css_class(decision: str) -> str:
    """Map a decision string to a CSS class for the banner."""
    mapping = {
        "Strong Buy": "decision-strong-buy",
        "Buy": "decision-buy",
        "Sell": "decision-sell",
        "Strong Sell": "decision-strong-sell",
        "No Trade": "decision-no-trade",
        "Wait": "decision-wait",
    }
    return mapping.get(decision, "decision-no-trade")


def _display_decision_banner(decision: str, confidence: float) -> None:
    """Render the large decision banner."""
    css_class = _get_decision_css_class(decision)
    st.markdown(
        f'<div class="decision-banner {css_class}">'
        f'{decision}'
        f'<br><span style="font-size:1rem;font-weight:400;">'
        f'Confidence: {confidence:.1f}%</span></div>',
        unsafe_allow_html=True,
    )


def _display_metrics(result: FullAnalysisResult) -> None:
    """Render the key metrics row."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Technical Score", f"{result.technical_score:.1f}/100")
    with col2:
        st.metric("Correlation Score", f"{result.correlation_score:.1f}/100")
    with col3:
        st.metric("Combined Score", f"{result.combined_score:.1f}/100")
    with col4:
        st.metric("Session", result.session)


def _display_trade_params(result: FullAnalysisResult) -> None:
    """Render entry, SL, TP, R:R metrics."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Entry", f"{result.entry:.5f}")
    with col2:
        st.metric("Stop Loss", f"{result.stop_loss:.5f}")
    with col3:
        st.metric("Take Profit", f"{result.take_profit:.5f}")
    with col4:
        st.metric("Risk : Reward", f"1 : {result.risk_reward:.1f}")


def _display_chart(result: FullAnalysisResult) -> None:
    """Render the candlestick chart with indicator overlays."""
    if result.technical_result and result.technical_result.raw:
        ta = result.technical_result.raw
        if ta.df is not None and not ta.df.empty:
            fig = create_candlestick_chart(
                df=ta.df,
                pair=result.pair,
                ema20=ta.ema20_series,
                ema50=ta.ema50_series,
                ema200=ta.ema200_series,
                support_levels=ta.support_levels,
                resistance_levels=ta.resistance_levels,
                entry=result.entry if result.entry > 0 else None,
                sl=result.stop_loss if result.stop_loss > 0 else None,
                tp=result.take_profit if result.take_profit > 0 else None,
            )
            st.plotly_chart(fig, use_container_width=True)


def _display_correlation(result: FullAnalysisResult) -> None:
    """Render correlation check results."""
    if not result.correlation_result:
        st.info("Correlation analysis disabled")
        return

    corr = result.correlation_result
    st.subheader("Correlation Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Score:** {corr.correlation_score:.1f}/100")
        st.markdown(f"**Confidence Modifier:** {corr.confidence_modifier:+.1f}")
        st.markdown(f"**Details:** {corr.details}")

    with col2:
        if corr.checks:
            scores = {c["pair"]: c["score"] for c in corr.checks}
            fig = create_correlation_bar(scores)
            st.plotly_chart(fig, use_container_width=True)

    # Table of checks
    if corr.checks:
        check_df = pd.DataFrame(corr.checks)
        st.dataframe(
            check_df.style.format({"score": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )


def _display_multi_timeframe(result: FullAnalysisResult) -> None:
    """Render multi-timeframe analysis."""
    if not result.multi_timeframe:
        return

    mtf = result.multi_timeframe
    st.subheader("Multi-Timeframe Analysis")
    st.markdown(f"**Alignment:** `{mtf['alignment']}`")

    if mtf["timeframes"]:
        for tf_name, tf_info in mtf["timeframes"].items():
            st.markdown(f"- **{tf_name}**: {tf_info}")


def _display_news(result: FullAnalysisResult) -> None:
    """Render news analysis results."""
    if not result.news_result:
        st.info("News filter disabled")
        return

    news = result.news_result
    st.subheader("News Filter")

    col1, col2, col3 = st.columns(3)
    with col1:
        status_emoji = "🟢" if news.status == "clear" else ("🟡" if news.status == "caution" else "🔴")
        st.markdown(f"**Status:** {status_emoji} {news.status.upper()}")
    with col2:
        st.markdown(f"**Score:** {news.news_score:.1f}/100")
    with col3:
        st.markdown(f"**Events Found:** {news.event_count}")

    st.markdown(f"*{news.details}*")

    if news.events:
        with expander("View Events"):
            event_df = pd.DataFrame(news.events)
            if not event_df.empty:
                st.dataframe(event_df, use_container_width=True, hide_index=True)


def _display_individual_scores(result: FullAnalysisResult) -> None:
    """Render individual indicator score gauges."""
    if not result.technical_result:
        return

    scores = result.technical_result.individual_scores
    if not scores:
        return

    st.subheader("Individual Indicator Scores")

    gauge_cols = st.columns(4)
    items = list(scores.items())
    for i, (name, score) in enumerate(items):
        with gauge_cols[i % 4]:
            fig = create_score_gauge(score, title=name)
            st.plotly_chart(fig, use_container_width=True)


def _display_ai_reasoning(result: FullAnalysisResult) -> None:
    """Render the AI reasoning section."""
    if not result.ai_reasoning:
        return

    st.subheader("AI Reasoning")
    st.markdown(
        f'<div class="reasoning-box">{result.ai_reasoning}</div>',
        unsafe_allow_html=True,
    )

    if result.prompt_result:
        st.caption(f"Model: {result.prompt_result.model_used}")


def _display_backtest(db: TradeDatabase) -> None:
    """Render the backtest section."""
    st.subheader("Backtest & Performance")

    trades = db.get_all_trades()
    if not trades:
        st.info("No trades recorded yet. Run an analysis to start building history.")
        return

    bt = compute_backtest(trades)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", bt.total_trades)
        st.metric("Win Rate", f"{bt.win_rate:.1f}%")
    with col2:
        st.metric("Loss Rate", f"{bt.loss_rate:.1f}%")
        st.metric("Profit Factor", f"{bt.profit_factor:.2f}")
    with col3:
        st.metric("Max Drawdown", f"{bt.max_drawdown:.5f}")
        st.metric("Sharpe Ratio", f"{bt.sharpe_ratio:.2f}")
    with col4:
        st.metric("Total Profit", f"{bt.total_profit:.5f}")
        st.metric("Avg Profit/Trade", f"{bt.avg_profit:.5f}")

    # Trade history table
    with expander("Trade History"):
        if trades:
            trade_df = pd.DataFrame(trades)
            trade_df = trade_df.sort_values("id", ascending=False)
            st.dataframe(
                trade_df[[
                    "timestamp", "pair", "timeframe", "decision",
                    "confidence", "entry_price", "stop_loss",
                    "take_profit", "outcome", "profit_loss",
                ]].head(50),
                use_container_width=True,
                hide_index=True,
            )


def _display_memory(result: FullAnalysisResult) -> None:
    """Render memory/learning section."""
    if not result.memory_result:
        return

    mem = result.memory_result
    st.subheader("AI Memory")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Trades Evaluated:** {mem.total_evaluated}")
        st.markdown(f"**Correct Signals:** {mem.correct_signals}")
        st.markdown(f"**Incorrect Signals:** {mem.incorrect_signals}")
        st.markdown(f"**Accuracy:** {mem.accuracy:.1f}%")
    with col2:
        if mem.weight_adjustments:
            st.markdown("**Adapted Weights:**")
            for name, weight in mem.weight_adjustments.items():
                st.markdown(f"- {name}: {weight:.1f}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> dict:
    """Render the sidebar controls and return user selections.

    Returns:
        Dictionary with all user-selected parameters.
    """
    with st.sidebar:
        st.title("AI Forex Agent")
        st.markdown("---")

        # Pair selection — only pairs that have correlation data
        available_pairs = sorted(set(ALL_PAIRS))
        pair = st.selectbox(
            label="Currency Pair",
            options=available_pairs,
            index=0 if available_pairs else 0,
            key="pair_select",
        )

        # Timeframe
        tf_options = [tf.value for tf in Timeframe]
        timeframe = st.selectbox(
            label="Timeframe",
            options=tf_options,
            index=2,  # Default H1
            key="tf_select",
        )

        # AI Model
        model_options = [m.value for m in AIModel]
        ai_model = st.selectbox(
            label="AI Model",
            options=model_options,
            index=0,
            key="model_select",
        )

        st.markdown("---")
        st.subheader("Filters & Indicators")

        use_correlation = st.checkbox("Use Correlation Filter", value=True, key="chk_corr")
        use_news = st.checkbox("Use News Filter", value=True, key="chk_news")
        use_multi_tf = st.checkbox("Use Multi Timeframe", value=True, key="chk_mtf")
        use_candlestick = st.checkbox("Use Candlestick Pattern", value=True, key="chk_candle")
        use_ai_reasoning = st.checkbox("Use AI Reasoning", value=True, key="chk_ai")

        st.markdown("---")
        st.subheader("Indicators")

        use_atr = st.checkbox("Use ATR", value=True, key="chk_atr")
        use_sr = st.checkbox("Use Support Resistance", value=True, key="chk_sr")
        use_ema = st.checkbox("Use EMA", value=True, key="chk_ema")
        use_macd = st.checkbox("Use MACD", value=True, key="chk_macd")
        use_rsi = st.checkbox("Use RSI", value=True, key="chk_rsi")
        use_adx = st.checkbox("Use ADX", value=True, key="chk_adx")

        st.markdown("---")
        st.markdown(
            """
            **AI Forex Decision Agent** v1.0

            Multi-agent architecture with:
            - 8 AI Agents
            - Correlation Engine
            - News Filter
            - AI Reasoning (GPT)
            - Risk Management 1:2
            - AI Memory & Learning
            """
        )

    return {
        "pair": pair,
        "timeframe": timeframe,
        "ai_model": ai_model,
        "use_correlation": use_correlation,
        "use_news": use_news,
        "use_multi_tf": use_multi_tf,
        "use_candlestick": use_candlestick,
        "use_ai_reasoning": use_ai_reasoning,
        "use_atr": use_atr,
        "use_sr": use_sr,
        "use_ema": use_ema,
        "use_macd": use_macd,
        "use_rsi": use_rsi,
        "use_adx": use_adx,
    }


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Streamlit application."""

    # Render sidebar
    params = _render_sidebar()

    # Database
    db_path = get_db_path()
    db = TradeDatabase(db_path)

    # Header
    st.title("AI Forex Decision Agent")
    st.caption(
        f"{params['pair']} | {params['timeframe']} | "
        f"Session: {st.session_state.get('session_text', '—')}"
    )

    # Analyse button
    analyse_clicked = st.button(
        label="ANALISA SEKARANG",
        type="primary",
        use_container_width=True,
        key="btn_analyse",
    )

    if analyse_clicked:
        # Resolve timeframe enum
        try:
            tf_enum = Timeframe(params["timeframe"])
        except ValueError:
            tf_enum = Timeframe.H1

        # Resolve AI model
        try:
            model_enum = AIModel(params["ai_model"])
        except ValueError:
            model_enum = AIModel.GPT4O_MINI

        # Build the decision agent
        agent = DecisionAgent(
            model=model_enum,
            use_correlation=params["use_correlation"],
            use_news=params["use_news"],
            use_multi_tf=params["use_multi_tf"],
            use_candlestick=params["use_candlestick"],
            use_ai_reasoning=params["use_ai_reasoning"],
            use_atr=params["use_atr"],
            use_sr=params["use_sr"],
            use_ema=params["use_ema"],
            use_macd=params["use_macd"],
            use_rsi=params["use_rsi"],
            use_adx=params["use_adx"],
            db=db,
        )

        # Run the analysis with a progress indicator
        with st.spinner(
            f"Analysing {params['pair']} @ {params['timeframe']}... "
            "This may take a moment."
        ):
            try:
                result = agent.run(params["pair"], tf_enum)
                st.session_state["last_result"] = result
                st.session_state["analysis_done"] = True
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logger.error("Analysis failed: %s", e, exc_info=True)
                st.session_state["analysis_done"] = False

    # Display results if analysis has been run
    if st.session_state.get("analysis_done"):
        result: FullAnalysisResult = st.session_state["last_result"]

        # Update session display
        st.caption(
            f"{result.pair} | {result.timeframe} | Session: {result.session}"
        )

        # --- Tab Layout ---
        tab_main, tab_details, tab_backtest = st.tabs([
            "Dashboard", "Detailed Analysis", "Backtest",
        ])

        with tab_main:
            # Decision banner
            _display_decision_banner(result.decision, result.confidence)

            # Key metrics
            _display_metrics(result)

            st.markdown("---")

            # Trade parameters
            _display_trade_params(result)

            st.markdown("---")

            # Chart
            _display_chart(result)

            st.markdown("---")

            # AI Reasoning
            _display_ai_reasoning(result)

        with tab_details:
            # Individual scores
            _display_individual_scores(result)

            st.markdown("---")

            # Correlation
            _display_correlation(result)

            st.markdown("---")

            # Multi-timeframe
            _display_multi_timeframe(result)

            st.markdown("---")

            # News
            _display_news(result)

            st.markdown("---")

            # Memory
            _display_memory(result)

            st.markdown("---")

            # Trend details
            if result.trend_result:
                st.subheader("Trend Analysis")
                st.markdown(f"**Direction:** {result.trend_result.direction}")
                st.markdown(f"**Strength:** {result.trend_result.strength}")
                st.markdown(f"**EMA Alignment:** {result.trend_result.ema_alignment}")
                st.markdown(f"**Higher TF Trend:** {result.trend_result.higher_tf_trend}")
                st.markdown(f"**Trend Score:** {result.trend_result.score:.1f}/100")

        with tab_backtest:
            _display_backtest(db)

    # Initial state
    elif not analyse_clicked:
        st.markdown(
            """
            ## Welcome to AI Forex Decision Agent

            Select a currency pair and timeframe from the sidebar,
            configure your preferred indicators and filters,
            then click **ANALISA SEKARANG** to run the analysis.

            The AI will:
            1. Analyse trend using EMA alignment
            2. Compute all technical indicators
            3. Check correlation across related pairs
            4. Filter for high-impact news
            5. Perform multi-timeframe analysis
            6. Calculate risk management parameters
            7. Send all data to GPT for reasoning
            8. Produce a final BUY / SELL / NO TRADE decision
            """
        )
        # Show existing trade count
        try:
            trades = db.get_trades(limit=1)
            if trades:
                st.caption(f"Database contains existing trade records. Check the Backtest tab.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
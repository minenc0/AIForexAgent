"""Charting utilities — Plotly candlestick with indicator overlays."""

from __future__ import annotations

from typing import Any, Optional

import plotly.graph_objects as go
import pandas as pd


def create_candlestick_chart(
    df: pd.DataFrame,
    pair: str,
    ema20: Optional[pd.Series] = None,
    ema50: Optional[pd.Series] = None,
    ema200: Optional[pd.Series] = None,
    support_levels: Optional[list[float]] = None,
    resistance_levels: Optional[list[float]] = None,
    entry: Optional[float] = None,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
) -> go.Figure:
    """Build a Plotly candlestick chart with technical overlays.

    Args:
        df: OHLCV DataFrame with columns ``Open, High, Low, Close, Volume``.
        pair: Currency pair label for the title.
        ema20: EMA-20 series aligned to ``df`` index.
        ema50: EMA-50 series aligned to ``df`` index.
        ema200: EMA-200 series aligned to ``df`` index.
        support_levels: List of horizontal support prices.
        resistance_levels: List of horizontal resistance prices.
        entry: Entry price marker.
        sl: Stop-loss price marker.
        tp: Take-profit price marker.

    Returns:
        A ``plotly.graph_objects.Figure`` ready for ``st.plotly_chart``.
    """
    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#00C853",
            decreasing_line_color="#FF1744",
        )
    )

    # EMAs
    if ema20 is not None:
        fig.add_trace(
            go.Scatter(x=ema20.index, y=ema20, name="EMA 20",
                       line=dict(color="#FF9800", width=1.2))
        )
    if ema50 is not None:
        fig.add_trace(
            go.Scatter(x=ema50.index, y=ema50, name="EMA 50",
                       line=dict(color="#2196F3", width=1.2))
        )
    if ema200 is not None:
        fig.add_trace(
            go.Scatter(x=ema200.index, y=ema200, name="EMA 200",
                       line=dict(color="#E91E63", width=1.2))
        )

    # Support / Resistance horizontal lines
    if support_levels:
        for i, level in enumerate(support_levels):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="#4CAF50",
                annotation_text=f"S{i+1} {level:.5f}",
                annotation_position="top left",
            )
    if resistance_levels:
        for i, level in enumerate(resistance_levels):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="#F44336",
                annotation_text=f"R{i+1} {level:.5f}",
                annotation_position="bottom left",
            )

    # Entry / SL / TP markers
    last_idx = df.index[-1]
    if entry is not None:
        fig.add_hline(y=entry, line_color="#FFFFFF", line_width=1.5,
                       annotation_text=f"Entry {entry:.5f}",
                       annotation_position="top right")
    if sl is not None:
        fig.add_hline(y=sl, line_color="#FF1744", line_width=2,
                       annotation_text=f"SL {sl:.5f}",
                       annotation_position="bottom right")
    if tp is not None:
        fig.add_hline(y=tp, line_color="#00E676", line_width=2,
                       annotation_text=f"TP {tp:.5f}",
                       annotation_position="top right")

    fig.update_layout(
        title=f"{pair} — Technical Analysis",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        margin=dict(l=60, r=60, t=40, b=40),
        xaxis=dict(
            gridcolor="#333",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="#333",
            tickfont=dict(size=10),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )
    return fig


def create_score_gauge(score: float, title: str = "Technical Score") -> go.Figure:
    """Create a gauge chart for a numeric score.

    Args:
        score: Value between 0 and 100.
        title: Gauge title.

    Returns:
        A Plotly gauge figure.
    """
    color = "#FF1744"
    if score >= 80:
        color = "#00E676"
    elif score >= 60:
        color = "#FFEB3B"
    elif score >= 40:
        color = "#FF9800"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"size": 16, "color": "#FAFAFA"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#FAFAFA"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 20], "color": "#B71C1C"},
                    {"range": [20, 40], "color": "#E65100"},
                    {"range": [40, 60], "color": "#F9A825"},
                    {"range": [60, 80], "color": "#2E7D32"},
                    {"range": [80, 100], "color": "#1B5E20"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=30, r=30, t=40, b=20),
    )
    return fig


def create_correlation_bar(
    scores: dict[str, float],
    title: str = "Correlation Score",
) -> go.Figure:
    """Create a horizontal bar chart of correlation scores.

    Args:
        scores: Mapping of pair name to score (0-100).
        title: Chart title.

    Returns:
        A Plotly bar figure.
    """
    pairs = list(scores.keys())
    values = list(scores.values())
    colors = ["#00E676" if v >= 70 else "#FF9800" if v >= 40 else "#FF1744" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=pairs,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=300,
        margin=dict(l=80, r=40, t=40, b=20),
        xaxis=dict(range=[0, 110], title="Score"),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig
"""News Analysis Agent — filters trades based on upcoming economic events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from data.economic_calendar import (
    fetch_forex_factory_calendar,
    get_news_score,
    has_high_impact_news,
    get_current_session,
    NewsEvent,
)
from utils.logger import logger


@dataclass
class NewsResult:
    """Result of the news analysis."""

    has_high_impact: bool
    news_score: float  # 0-100
    event_count: int
    high_impact_events: list[str]
    status: str  # 'clear', 'caution', 'no_trade'
    details: str
    events: list[dict]  # For display


class NewsAgent:
    """Analyses upcoming economic events and determines if it is safe to trade.

    If a high-impact event is within ±30 minutes, the agent returns a
    NO TRADE status.
    """

    def __init__(self, window_minutes: int = 30) -> None:
        """Initialise the NewsAgent.

        Args:
            window_minutes: Minutes before/after an event to block trading.
        """
        self.window_minutes = window_minutes

    def analyse(
        self,
        pair: str,
        currencies: Optional[list[str]] = None,
    ) -> NewsResult:
        """Run the news analysis for a given pair.

        Args:
            pair: Forex pair (e.g. ``'EURUSD'``). Used to extract
                relevant currencies.
            currencies: Override currency filter. If ``None``, extracted
                from the pair.

        Returns:
            A ``NewsResult`` with the analysis outcome.
        """
        if currencies is None:
            currencies = [pair[:3], pair[3:6]]  # e.g. ['EUR', 'USD']

        logger.info(
            "NewsAgent analysing for %s (currencies: %s)", pair, currencies
        )

        result = NewsResult(
            has_high_impact=False,
            news_score=100.0,
            event_count=0,
            high_impact_events=[],
            status="clear",
            details="",
            events=[],
        )

        all_events: list[dict] = []
        all_news_events: list[NewsEvent] = []

        for currency in currencies:
            events = fetch_forex_factory_calendar(currency)
            all_news_events.extend(events)

        # Deduplicate by title
        seen_titles: set[str] = set()
        unique_events: list[NewsEvent] = []
        for e in all_news_events:
            if e.title not in seen_titles:
                seen_titles.add(e.title)
                unique_events.append(e)

        result.event_count = len(unique_events)

        for event in unique_events:
            all_events.append({
                "datetime": event.datetime,
                "currency": event.currency,
                "impact": event.impact,
                "title": event.title,
                "forecast": event.forecast,
                "previous": event.previous,
            })
            if event.impact == "High":
                result.high_impact_events.append(
                    f"{event.currency}: {event.title}"
                )

        result.events = all_events

        # Check for high-impact news in window
        result.has_high_impact = has_high_impact_news(
            unique_events, self.window_minutes
        )

        # Calculate score
        result.news_score = get_news_score(
            unique_events, self.window_minutes
        )

        if result.has_high_impact:
            result.status = "no_trade"
            result.details = (
                f"BLOCKED: High-impact news within {self.window_minutes} minutes. "
                f"Events: {', '.join(result.high_impact_events[:3])}"
            )
        elif result.news_score < 50:
            result.status = "caution"
            result.details = (
                f"CAUTION: Medium-impact news nearby. "
                f"Total events: {result.event_count}"
            )
        else:
            result.status = "clear"
            result.details = (
                f"Clear of high-impact news. "
                f"Total events in calendar: {result.event_count}"
            )

        session = get_current_session()
        result.details += f" | Session: {session}"

        logger.info("NewsAgent result: %s", result.details)
        return result
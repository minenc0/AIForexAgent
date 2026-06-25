"""Economic calendar / news filter — Forex Factory scraping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

from utils.logger import logger

# Session timezone offsets (UTC hours)
SESSION_TIMES: dict[str, tuple[int, int]] = {
    "Sydney": (21, 6),     # 21:00 UTC - 06:00 UTC
    "Tokyo": (0, 9),       # 00:00 UTC - 09:00 UTC
    "London": (7, 16),     # 07:00 UTC - 16:00 UTC
    "New York": (12, 21),  # 12:00 UTC - 21:00 UTC
}


@dataclass
class NewsEvent:
    """A single economic calendar event."""

    datetime: str
    currency: str
    impact: str
    title: str
    forecast: str = ""
    previous: str = ""


def get_current_session(utc_now: Optional[datetime] = None) -> str:
    """Determine which forex session is currently active.

    Args:
        utc_now: Override for testing. Defaults to current UTC time.

    Returns:
        Session name (e.g. ``'London'``, ``'New York'``).
    """
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)

    hour = utc_now.hour

    sessions: list[str] = []
    for name, (start, end) in SESSION_TIMES.items():
        if start < end:
            if start <= hour < end:
                sessions.append(name)
        else:  # Wraps midnight (e.g. Sydney 21-6)
            if hour >= start or hour < end:
                sessions.append(name)

    if not sessions:
        return "Off-Session"
    return ", ".join(sessions)


def fetch_forex_factory_calendar(
    currency: Optional[str] = None,
) -> list[NewsEvent]:
    """Scrape the Forex Factory economic calendar.

    Fetches upcoming news events and filters by currency if provided.

    Args:
        currency: ISO currency code to filter (e.g. ``'USD'``, ``'EUR'``).
            If ``None``, returns all events.

    Returns:
        A list of ``NewsEvent`` objects.
    """
    url = "https://www.forexfactory.com/calendar"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to fetch Forex Factory calendar: %s", e)
        return []

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        events: list[NewsEvent] = []

        rows = soup.select("tr.calendar__row")
        for row in rows:
            if "calendar__row--new-day" in row.get("class", []):
                continue

            date_el = row.select_one("td.calendar__date")
            time_el = row.select_one("td.calendar__time")
            currency_el = row.select_one("td.calendar__currency span")
            impact_el = row.select_one("td.calendar__impact")
            title_el = row.select_one("td.calendar__event")
            forecast_el = row.select_one("td.calendar__forecast")
            previous_el = row.select_one("td.calendar__previous")

            if not title_el:
                continue

            event_date = date_el.get_text(strip=True) if date_el else ""
            event_time = time_el.get_text(strip=True) if time_el else ""
            event_currency = currency_el.get_text(strip=True) if currency_el else ""
            event_title = title_el.get_text(strip=True) if title_el else ""
            event_forecast = forecast_el.get_text(strip=True) if forecast_el else ""
            event_previous = previous_el.get_text(strip=True) if previous_el else ""

            # Impact level
            impact = "Low"
            if impact_el:
                impact_icons = impact_el.select("span")
                if len(impact_icons) >= 3:
                    impact = "High"
                elif len(impact_icons) >= 2:
                    impact = "Medium"

            events.append(NewsEvent(
                datetime=f"{event_date} {event_time}",
                currency=event_currency,
                impact=impact,
                title=event_title,
                forecast=event_forecast,
                previous=event_previous,
            ))

        logger.info("Fetched %d news events from Forex Factory", len(events))

        if currency:
            events = [e for e in events if e.currency == currency]
            logger.info("Filtered to %d events for %s", len(events), currency)

        return events

    except Exception as e:
        logger.error("Error parsing Forex Factory calendar: %s", e)
        return []


def has_high_impact_news(
    events: list[NewsEvent],
    window_minutes: int = 30,
    utc_now: Optional[datetime] = None,
) -> bool:
    """Check if there is a high-impact event within the time window.

    Args:
        events: List of news events.
        window_minutes: Minutes before/after current time to check.
        utc_now: Override for testing.

    Returns:
        ``True`` if a high-impact event is within the window.
    """
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)

    for event in events:
        if event.impact != "High":
            continue
        try:
            # Parse the event datetime — format varies on Forex Factory
            # Use a simplified approach: check if event string contains time
            dt_str = event.datetime.strip()
            if not dt_str or not any(c.isdigit() for c in dt_str):
                continue

            # For robustness, we check the time portion
            parts = dt_str.split()
            time_part = ""
            for p in parts:
                if ":" in p:
                    time_part = p
                    break

            if not time_part:
                continue

            # Parse time (handle am/pm or 24h format)
            time_part = time_part.replace("a", "").replace("m", "").replace("p", "").strip()
            h_m = time_part.split(":")
            if len(h_m) < 2:
                continue

            hour = int(h_m[0])
            minute = int(h_m[1])

            # Assume today for comparison
            event_dt = utc_now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)

            diff = abs((event_dt - utc_now).total_seconds()) / 60.0
            if diff <= window_minutes:
                logger.warning(
                    "High impact news within %d min: %s (%s)",
                    window_minutes, event.title, event.currency,
                )
                return True

        except (ValueError, IndexError) as e:
            logger.debug("Could not parse event datetime '%s': %s", event.datetime, e)
            continue

    return False


def get_news_score(
    events: list[NewsEvent],
    window_minutes: int = 30,
) -> float:
    """Calculate a news-based score.

    High-impact news within the window → score 0 (NO TRADE signal).
    Medium-impact within window → score 30.
    No relevant news → score 100.

    Args:
        events: List of news events.
        window_minutes: Time window in minutes.

    Returns:
        A score from 0 to 100.
    """
    if has_high_impact_news(events, window_minutes):
        return 0.0

    # Check for medium-impact events
    utc_now = datetime.now(timezone.utc)
    for event in events:
        if event.impact != "Medium":
            continue
        try:
            dt_str = event.datetime.strip()
            parts = dt_str.split()
            time_part = ""
            for p in parts:
                if ":" in p:
                    time_part = p
                    break
            if not time_part:
                continue
            time_part = time_part.replace("a", "").replace("m", "").replace("p", "").strip()
            h_m = time_part.split(":")
            if len(h_m) < 2:
                continue
            hour = int(h_m[0])
            minute = int(h_m[1])
            event_dt = utc_now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
            diff = abs((event_dt - utc_now).total_seconds()) / 60.0
            if diff <= window_minutes:
                return 30.0
        except (ValueError, IndexError):
            continue

    return 100.0
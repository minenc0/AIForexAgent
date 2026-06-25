"""Configuration loader — reads .env and builds runtime config."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from config import AIModel, AppConfig, Timeframe

load_dotenv()


def get_openai_api_key() -> str:
    """Return the OpenAI API key from environment variables.

    Raises:
        ValueError: If the key is not set or is a placeholder.
    """
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-your"):
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Set it in .env or as an environment variable."
        )
    return key


def get_ai_model(model_name: Optional[str] = None) -> AIModel:
    """Resolve an AI model enum from a string name.

    Args:
        model_name: String representation (e.g. ``'gpt-4o'``). Falls back
            to the environment variable ``AI_MODEL`` then to ``gpt-4o-mini``.

    Returns:
        An ``AIModel`` enum member.
    """
    name = model_name or os.getenv("AI_MODEL", "gpt-4o-mini")
    try:
        return AIModel(name)
    except ValueError:
        return AIModel.GPT4O_MINI


def get_default_timeframe() -> Timeframe:
    """Return the default timeframe from env or fallback to H1."""
    tf = os.getenv("DEFAULT_TIMEFRAME", "1h")
    try:
        return Timeframe(tf)
    except ValueError:
        return Timeframe.H1


def get_project_root() -> Path:
    """Return the project root directory (where this file's parent lives)."""
    return Path(__file__).resolve().parent.parent


def get_db_path() -> Path:
    """Return the SQLite database file path."""
    return get_project_root() / "database" / "trade_history.db"


def build_app_config(
    model_name: Optional[str] = None,
) -> AppConfig:
    """Build a fully-resolved ``AppConfig`` from env / explicit params.

    Args:
        model_name: Override AI model name.

    Returns:
        A ready-to-use ``AppConfig``.
    """
    return AppConfig(
        ai_model=get_ai_model(model_name),
        default_timeframe=get_default_timeframe(),
    )
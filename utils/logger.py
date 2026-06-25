"""Centralised logging configuration for AI Forex Decision Agent."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "ai_forex_agent",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create and return a configured logger instance.

    Args:
        name: Logger name.
        level: Logging level (default INFO).
        log_file: Optional file path to persist logs.

    Returns:
        A configured ``logging.Logger``.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Module-level singleton
logger = setup_logger()
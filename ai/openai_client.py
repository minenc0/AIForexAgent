"""OpenAI API client with retry, error handling, and token budgeting."""

from __future__ import annotations

import time
from typing import Any, Optional

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

from utils.logger import logger
from utils.config_loader import get_openai_api_key


def get_openai_client() -> OpenAI:
    """Create and return an OpenAI client instance.

    Reads the API key from the environment.

    Returns:
        An ``openai.OpenAI`` client.

    Raises:
        ValueError: If the API key is not configured.
    """
    api_key = get_openai_api_key()
    return OpenAI(api_key=api_key)


def chat_completion(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    backoff: float = 2.0,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Send a chat completion request to the OpenAI API with retry logic.

    Args:
        model: Model identifier (e.g. ``'gpt-4o-mini'``).
        system_prompt: The system message establishing the AI's role.
        user_prompt: The user message with the analysis data.
        max_retries: Maximum number of retry attempts.
        backoff: Exponential backoff base in seconds.
        temperature: Sampling temperature (low for deterministic output).
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant's response text.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    client = get_openai_client()

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "OpenAI request: model=%s attempt=%d/%d",
                model, attempt, max_retries,
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            if not content:
                logger.warning("OpenAI returned empty response")
                continue

            tokens_used = response.usage.total_tokens if response.usage else 0
            logger.info("OpenAI response received (%d tokens)", tokens_used)
            return content.strip()

        except RateLimitError as e:
            logger.warning("Rate limited (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(backoff ** attempt + 1)

        except APITimeoutError as e:
            logger.warning("API timeout (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(backoff ** attempt)

        except APIError as e:
            logger.error("API error (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(backoff ** attempt)

        except Exception as e:
            logger.error("Unexpected error calling OpenAI (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(backoff ** attempt)

    raise RuntimeError(
        f"OpenAI API failed after {max_retries} attempts for model {model}"
    )
"""
Shared Groq client wrapper. ALL modules must import from here — never call the
Groq SDK directly anywhere else in this codebase.

Features:
- Reads GROQ_API_KEY from environment (raises EnvironmentError if absent)
- Exponential backoff retry on 429 RateLimitError (max 3 retries)
- Request queue enforcing 2-second minimum gap between calls (≤ 30 rpm)
- Per-call token usage logging to stdout
- Hard limit guard: raises RateLimitWarning if session input tokens exceed
  5000 within a rolling 60-second window
"""

import os
import time
import logging
import threading
from collections import deque
from typing import List, Dict, Any

import groq

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-level token rate guard
# ---------------------------------------------------------------------------

_TOKEN_WINDOW_SECONDS = 60
_TOKEN_WINDOW_LIMIT = 5000

class RateLimitWarning(Exception):
    """Raised before a Groq call when the session would exceed the token budget."""


class _TokenGuard:
    """Tracks input tokens over a rolling 60-second window, thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        # Each entry: (timestamp, input_tokens)
        self._window: deque = deque()

    def check_and_record(self, input_tokens: int) -> None:
        now = time.monotonic()
        with self._lock:
            # Evict entries older than the window
            while self._window and now - self._window[0][0] > _TOKEN_WINDOW_SECONDS:
                self._window.popleft()
            total = sum(t for _, t in self._window) + input_tokens
            if total > _TOKEN_WINDOW_LIMIT:
                raise RateLimitWarning(
                    f"Session input token budget exceeded: {total} tokens in the last "
                    f"{_TOKEN_WINDOW_SECONDS}s (limit {_TOKEN_WINDOW_LIMIT}). "
                    "Wait before sending more requests."
                )
            self._window.append((now, input_tokens))


_token_guard = _TokenGuard()

# ---------------------------------------------------------------------------
# Request-gap throttle (2-second minimum between calls)
# ---------------------------------------------------------------------------

_MIN_GAP_SECONDS = 2.0
_last_call_time: float = 0.0
_call_lock = threading.Lock()


def _enforce_gap() -> None:
    global _last_call_time
    with _call_lock:
        now = time.monotonic()
        wait = _MIN_GAP_SECONDS - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.monotonic()


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

def _get_client() -> groq.Groq:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Export it before running any Groq-backed module."
        )
    return groq.Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def groq_chat(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Send a chat completion request to Groq with retry, throttle, and token tracking.

    Returns a dict:
        {
            "content":       str,           # assistant message text
            "input_tokens":  int,
            "output_tokens": int,
            "model":         str,
            "latency_ms":    float,
        }

    Raises:
        EnvironmentError    – GROQ_API_KEY not set
        RateLimitWarning    – session token budget exceeded (pre-flight check)
        groq.RateLimitError – if all retries are exhausted after 429s
    """
    # Rough pre-flight token estimate: 4 chars ≈ 1 token
    estimated_input = sum(len(m.get("content", "")) for m in messages) // 4
    _token_guard.check_and_record(estimated_input)

    client = _get_client()

    MAX_RETRIES = 3
    base_delay = 2.0  # seconds

    for attempt in range(MAX_RETRIES + 1):
        _enforce_gap()
        t0 = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except groq.RateLimitError as exc:
            if attempt == MAX_RETRIES:
                log.error("Groq RateLimitError after %d retries: %s", MAX_RETRIES, exc)
                raise
            delay = base_delay * (2 ** attempt)
            log.warning(
                "Groq 429 RateLimitError (attempt %d/%d). Retrying in %.1fs.",
                attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)
            continue
        except groq.APITimeoutError as exc:
            if attempt == MAX_RETRIES:
                raise
            delay = base_delay * (2 ** attempt)
            log.warning("Groq timeout (attempt %d/%d). Retrying in %.1fs.", attempt + 1, MAX_RETRIES, delay)
            time.sleep(delay)
            continue

        latency_ms = (time.monotonic() - t0) * 1000

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        content = response.choices[0].message.content or ""

        log.info(
            "groq_chat | model=%s input_tokens=%d output_tokens=%d latency_ms=%.0f",
            model, input_tokens, output_tokens, latency_ms,
        )

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
            "latency_ms": latency_ms,
        }

    # Unreachable, but satisfies type checkers
    raise RuntimeError("groq_chat: exhausted retries without returning or raising")


def groq_stream(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
):
    """
    Stream a chat completion from Groq. Yields string content chunks as they arrive.

    Enforces the 2-second inter-request gap. Does not apply the token guard
    (token count is unknown until stream completes).
    """
    _enforce_gap()
    client = _get_client()
    t0 = time.monotonic()
    with client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    log.info("groq_stream | model=%s latency_ms=%.0f", model, (time.monotonic() - t0) * 1000)

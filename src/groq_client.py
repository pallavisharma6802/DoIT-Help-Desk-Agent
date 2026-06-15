import os
import time
import logging
import threading
from collections import deque
from typing import List, Dict, Any

import groq

log = logging.getLogger(__name__)

_TOKEN_WINDOW_SECONDS = 60
# Keep this comfortably below the hard provider limit so short bursts do not trip 429s.
_TOKEN_WINDOW_LIMIT = 14_000  # conservative ceiling; Groq free tier is 6k TPM per model

class RateLimitWarning(Exception):
    pass


class _TokenGuard:
    def __init__(self):
        self._lock = threading.Lock()
        # Store only successful calls in a rolling one-minute window.
        self._window: deque = deque()  # (timestamp, actual_tokens)

    def check(self, estimated_tokens: int) -> None:
        """Raise before the call if we'd clearly blow the budget."""
        now = time.monotonic()
        with self._lock:
            while self._window and now - self._window[0][0] > _TOKEN_WINDOW_SECONDS:
                self._window.popleft()
            total = sum(t for _, t in self._window) + estimated_tokens
            if total > _TOKEN_WINDOW_LIMIT:
                raise RateLimitWarning(
                    f"Token budget exceeded: {total} tokens in last {_TOKEN_WINDOW_SECONDS}s "
                    f"(limit {_TOKEN_WINDOW_LIMIT}). Wait before sending more requests."
                )

    def record(self, actual_tokens: int) -> None:
        """Record actual tokens used after a successful call."""
        with self._lock:
            self._window.append((time.monotonic(), actual_tokens))


_token_guard = _TokenGuard()

_MIN_GAP_SECONDS = 2.0
_last_call_time: float = 0.0
_call_lock = threading.Lock()


def _enforce_gap() -> None:
    global _last_call_time
    with _call_lock:
        now = time.monotonic()
        # Serialize callers and leave a small pause between requests to avoid bursty 429s.
        wait = _MIN_GAP_SECONDS - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.monotonic()



def _get_client() -> groq.Groq:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Export it before running any Groq-backed module."
        )
    return groq.Groq(api_key=api_key)


def groq_chat(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    # Estimate prompt size up front so we can fail fast before sending the request.
    estimated_input = sum(len(m.get("content", "")) for m in messages) // 4
    _token_guard.check(estimated_input)

    client = _get_client()

    # Retry only transient failures; daily quota exhaustion should surface immediately.
    MAX_RETRIES = 3
    base_delay = 2.0

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
            err_str = str(exc).lower()
            # Daily token limit (TPD) — retrying in seconds won't help
            if "tokens per day" in err_str or "per day" in err_str:
                log.error("Groq daily token limit reached: %s", exc)
                raise EnvironmentError(
                    "The AI service has reached its daily token limit. "
                    "Please try again tomorrow or contact the administrator."
                ) from exc
            # Per-minute rate limit (TPM) — retry with backoff
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

        # Record actual prompt usage, not the estimate, so the rolling window stays accurate.
        _token_guard.record(input_tokens)

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

    raise RuntimeError("groq_chat: exhausted retries without returning or raising")


def groq_stream(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.0,
):
    # Streaming still consumes prompt tokens up front, so it uses the same guard as chat.
    estimated_input = sum(len(m.get("content", "")) for m in messages) // 4
    _token_guard.check(estimated_input)
    _enforce_gap()
    client = _get_client()
    t0 = time.monotonic()
    total_output = 0
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
                # Streaming chunks do not expose final usage here, so approximate from text length.
                total_output += len(delta) // 4
                yield delta
    # Approximate total tokens so streamed calls also count against the minute window.
    _token_guard.record(estimated_input + total_output)
    log.info("groq_stream | model=%s latency_ms=%.0f", model, (time.monotonic() - t0) * 1000)

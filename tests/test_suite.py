"""
DoIT KB Agentic Assistant — test suite.
Never delete or skip a test. Add new tests only.

Run: pytest tests/test_suite.py -v
"""

import os
import sys
import time

import pytest

# Make src/ importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Environment: load .env if python-dotenv is available, otherwise rely on
# the shell environment having the keys set.
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # Keys must be set in the environment directly


# ===========================================================================
# T01 – T03: Classifier tests
# ===========================================================================

@pytest.fixture(scope="module")
def classifier():
    from classifier import classify_query
    return classify_query


class TestClassifier:

    def test_T01_simple_query(self, classifier):
        """T01: Simple query → complexity == 'simple'."""
        result = classifier("how do I reset my NetID password")
        assert result["complexity"] == "simple", (
            f"Expected 'simple', got {result['complexity']!r}. "
            f"Reasoning: {result['reasoning']}"
        )

    def test_T02_complex_query(self, classifier):
        """T02: Complex multi-condition query → complexity == 'complex'."""
        result = classifier(
            "my O365 stopped working after my affiliation changed "
            "and forwarding isn't set up"
        )
        assert result["complexity"] == "complex", (
            f"Expected 'complex', got {result['complexity']!r}. "
            f"Reasoning: {result['reasoning']}"
        )

    def test_T03_latency_under_500ms(self, classifier):
        """T03: Classifier (including retry overhead) returns in under 500ms
        on a non-rate-limited call measured wall-clock from the caller side.
        The groq_client 2-second gap enforcer applies between calls, so this
        test is run in isolation — the 2-second gap is between Groq requests,
        not between test runs. We measure only the classify_query() call itself
        which on a warm, non-rate-limited path should be well under 500ms for
        a 3B model on Groq's infrastructure.
        """
        # Brief sleep to ensure the 2-second inter-request gap in groq_client
        # is satisfied from the previous test so this call isn't queued.
        time.sleep(2.1)
        t0 = time.monotonic()
        result = classifier("how do I connect to eduroam WiFi")
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 500, (
            f"Classifier took {elapsed_ms:.0f}ms — exceeds 500ms budget. "
            "Check Groq free-tier latency or retry logic."
        )
        assert result["complexity"] in ("simple", "complex")

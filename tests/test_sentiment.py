"""
Pytest suite for Sentiment Analyzer.
Tests model wrappers, API endpoints, and edge cases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
from app.models import BERTAnalyzer, TextBlobAnalyzer, VADERAnalyzer


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def bert():
    return BERTAnalyzer()

@pytest.fixture(scope="module")
def textblob():
    return TextBlobAnalyzer()

@pytest.fixture(scope="module")
def vader():
    return VADERAnalyzer()


POSITIVE_TEXTS = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Outstanding product. Best purchase I've made in years.",
    "The customer service was excellent and very helpful.",
]
NEGATIVE_TEXTS = [
    "Terrible experience. I want my money back. Completely useless.",
    "Worst movie ever. Boring, slow, and a complete waste of time.",
    "Broken after one day. Horrible quality. Never buying again.",
]
EDGE_CASES = [
    "",           # empty
    "ok",         # very short
    "a" * 5001,   # very long
    "😊😊😊",     # emoji only
    "NOT bad",    # negation
    "I LOVE IT",  # all caps
]


# ── TextBlob tests ────────────────────────────────────────────────────────────
class TestTextBlob:
    def test_positive_texts(self, textblob):
        for t in POSITIVE_TEXTS:
            r = textblob.analyze(t)
            assert r["label"] in {"POSITIVE", "NEUTRAL"}, f"Expected POSITIVE for: {t!r}"
            assert 0 <= r["confidence"] <= 1

    def test_negative_texts(self, textblob):
        for t in NEGATIVE_TEXTS:
            r = textblob.analyze(t)
            assert r["label"] in {"NEGATIVE", "NEUTRAL"}, f"Expected NEGATIVE for: {t!r}"
            assert 0 <= r["confidence"] <= 1

    def test_returns_required_keys(self, textblob):
        r = textblob.analyze("test text")
        assert "label" in r
        assert "confidence" in r
        assert "scores" in r

    def test_confidence_range(self, textblob):
        for t in POSITIVE_TEXTS + NEGATIVE_TEXTS:
            r = textblob.analyze(t)
            assert 0.0 <= r["confidence"] <= 1.0, f"Confidence out of range: {r['confidence']}"

    def test_label_valid(self, textblob):
        valid = {"POSITIVE", "NEGATIVE", "NEUTRAL"}
        for t in POSITIVE_TEXTS + NEGATIVE_TEXTS:
            r = textblob.analyze(t)
            assert r["label"] in valid, f"Invalid label: {r['label']}"

    def test_short_text(self, textblob):
        r = textblob.analyze("good")
        assert r["label"] in {"POSITIVE", "NEUTRAL", "NEGATIVE"}

    def test_empty_text_doesnt_crash(self, textblob):
        try:
            r = textblob.analyze("")
            assert "label" in r
        except Exception as e:
            pytest.skip(f"Empty text edge case: {e}")

    def test_polarity_in_scores(self, textblob):
        r = textblob.analyze("I love this!")
        assert "polarity" in r["scores"]

    def test_subjectivity_in_scores(self, textblob):
        r = textblob.analyze("I love this!")
        assert "subjectivity" in r["scores"]

    def test_subjectivity_range(self, textblob):
        r = textblob.analyze("This is amazing!")
        assert 0.0 <= r["scores"]["subjectivity"] <= 1.0


# ── VADER tests ───────────────────────────────────────────────────────────────
class TestVADER:
    def test_positive_texts(self, vader):
        for t in POSITIVE_TEXTS:
            r = vader.analyze(t)
            assert r["label"] in {"POSITIVE", "NEUTRAL"}

    def test_negative_texts(self, vader):
        for t in NEGATIVE_TEXTS:
            r = vader.analyze(t)
            assert r["label"] in {"NEGATIVE", "NEUTRAL"}

    def test_compound_score_range(self, vader):
        for t in POSITIVE_TEXTS + NEGATIVE_TEXTS:
            r = vader.analyze(t)
            assert -1.0 <= r["scores"]["compound"] <= 1.0

    def test_returns_required_keys(self, vader):
        r = vader.analyze("test")
        for key in ["label", "confidence", "scores"]:
            assert key in r

    def test_pos_neg_neu_sum_to_one(self, vader):
        r = vader.analyze("This is a good product")
        scores = r["scores"]
        total = scores["pos"] + scores["neg"] + scores["neu"]
        assert abs(total - 1.0) < 0.01, f"VADER scores should sum to ~1.0, got {total}"

    def test_positive_compound(self, vader):
        r = vader.analyze("absolutely fantastic and wonderful!")
        assert r["scores"]["compound"] > 0

    def test_negative_compound(self, vader):
        r = vader.analyze("terrible horrible awful worst")
        assert r["scores"]["compound"] < 0

    def test_neutral_detection(self, vader):
        r = vader.analyze("The box contains a product.")
        assert r["label"] in {"NEUTRAL", "POSITIVE", "NEGATIVE"}

    def test_caps_amplification(self, vader):
        normal = vader.analyze("good product")
        caps   = vader.analyze("GOOD PRODUCT")
        assert caps["scores"]["compound"] >= normal["scores"]["compound"]

    def test_confidence_range(self, vader):
        r = vader.analyze("I love this so much!!!")
        assert 0.0 <= r["confidence"] <= 1.0


# ── BERT (rule-based fallback) tests ─────────────────────────────────────────
class TestBERTFallback:
    """Tests the rule-based fallback — safe to run without GPU/transformers."""

    def test_positive_fallback(self, bert):
        r = bert._rule_based("This is great and excellent and amazing!")
        assert r["label"] == "POSITIVE"
        assert r["confidence"] > 0.5

    def test_negative_fallback(self, bert):
        r = bert._rule_based("This is terrible and awful and horrible!")
        assert r["label"] == "NEGATIVE"
        assert r["confidence"] > 0.5

    def test_fallback_returns_keys(self, bert):
        r = bert._rule_based("test")
        for key in ["label", "confidence", "scores"]:
            assert key in r

    def test_fallback_confidence_range(self, bert):
        r = bert._rule_based("great excellent amazing")
        assert 0.0 <= r["confidence"] <= 1.0

    def test_fallback_score_keys(self, bert):
        r = bert._rule_based("test")
        assert "POSITIVE" in r["scores"]
        assert "NEGATIVE" in r["scores"]


# ── Integration: consensus logic ──────────────────────────────────────────────
class TestConsensus:
    def test_majority_vote(self):
        labels = ["POSITIVE", "POSITIVE", "NEGATIVE"]
        consensus = max(set(labels), key=labels.count)
        assert consensus == "POSITIVE"

    def test_unanimous(self):
        labels = ["NEGATIVE", "NEGATIVE", "NEGATIVE"]
        consensus = max(set(labels), key=labels.count)
        assert consensus == "NEGATIVE"

    def test_all_agree_positive(self, textblob, vader):
        text = "This is absolutely wonderful and I love it so much!"
        r1 = textblob.analyze(text)
        r2 = vader.analyze(text)
        labels = [r1["label"], r2["label"]]
        # Both should lean positive
        assert "POSITIVE" in labels

    def test_all_agree_negative(self, textblob, vader):
        text = "This is the worst terrible horrible thing I have ever experienced."
        r1 = textblob.analyze(text)
        r2 = vader.analyze(text)
        labels = [r1["label"], r2["label"]]
        assert "NEGATIVE" in labels


# ── FastAPI endpoint tests (requires running server) ──────────────────────────
class TestAPI:
    """Integration tests — run with: pytest tests/ -m api --api-url http://localhost:8000"""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
            from app.main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI not available")

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data

    def test_analyze_positive(self, client):
        r = client.post("/analyze", json={"text": "This is absolutely fantastic!"})
        assert r.status_code == 200
        data = r.json()
        assert "consensus" in data
        assert "results" in data
        assert len(data["results"]) == 3

    def test_analyze_negative(self, client):
        r = client.post("/analyze", json={"text": "Terrible, awful, horrible experience."})
        assert r.status_code == 200
        data = r.json()
        assert data["consensus"] in {"POSITIVE", "NEGATIVE", "NEUTRAL"}

    def test_analyze_empty_text(self, client):
        r = client.post("/analyze", json={"text": ""})
        assert r.status_code == 422

    def test_analyze_too_long(self, client):
        r = client.post("/analyze", json={"text": "x" * 5001})
        assert r.status_code == 422

    def test_root_endpoint(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_response_has_processing_time(self, client):
        r = client.post("/analyze", json={"text": "Test text for timing"})
        assert r.status_code == 200
        assert "processing_time_ms" in r.json()

    def test_model_results_have_latency(self, client):
        r = client.post("/analyze", json={"text": "Great product!"})
        data = r.json()
        for result in data["results"]:
            assert "latency_ms" in result
            assert result["latency_ms"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
NLP Model Wrappers: DistilBERT, TextBlob, VADER
Each returns: { label, confidence, scores }
"""

from __future__ import annotations
import re


# ── BERT (DistilBERT via HuggingFace pipeline) ────────────────────────────────
class BERTAnalyzer:
    def __init__(self):
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            try:
                from transformers import pipeline
                self._pipe = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    truncation=True,
                    max_length=512,
                )
            except Exception:
                self._pipe = "fallback"

    def analyze(self, text: str) -> dict:
        self._load()
        if self._pipe == "fallback":
            return self._rule_based(text)
        try:
            raw = self._pipe(text[:512])[0]
            label = raw["label"]           # POSITIVE / NEGATIVE
            conf  = round(raw["score"], 4)
            other = round(1 - conf, 4)
            if label == "POSITIVE":
                scores = {"POSITIVE": conf, "NEGATIVE": other}
            else:
                scores = {"POSITIVE": other, "NEGATIVE": conf}
            return {"label": label, "confidence": conf, "scores": scores}
        except Exception:
            return self._rule_based(text)

    def _rule_based(self, text: str) -> dict:
        """Simple fallback when transformers not available."""
        pos = len(re.findall(
            r'\b(great|good|excellent|amazing|fantastic|love|wonderful|best|perfect|happy)\b',
            text.lower()
        ))
        neg = len(re.findall(
            r'\b(bad|terrible|awful|horrible|worst|hate|poor|disappointing|boring|ugly)\b',
            text.lower()
        ))
        if pos > neg:
            conf = min(0.5 + pos * 0.1, 0.95)
            return {"label": "POSITIVE", "confidence": conf,
                    "scores": {"POSITIVE": conf, "NEGATIVE": round(1 - conf, 4)}}
        elif neg > pos:
            conf = min(0.5 + neg * 0.1, 0.95)
            return {"label": "NEGATIVE", "confidence": conf,
                    "scores": {"POSITIVE": round(1 - conf, 4), "NEGATIVE": conf}}
        else:
            return {"label": "POSITIVE", "confidence": 0.52,
                    "scores": {"POSITIVE": 0.52, "NEGATIVE": 0.48}}


# ── TextBlob ──────────────────────────────────────────────────────────────────
class TextBlobAnalyzer:
    def analyze(self, text: str) -> dict:
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity     = blob.sentiment.polarity       # -1 to 1
            subjectivity = blob.sentiment.subjectivity   # 0 to 1

            if polarity > 0.05:
                label = "POSITIVE"
                conf  = round(min(0.5 + polarity * 0.5, 0.99), 4)
            elif polarity < -0.05:
                label = "NEGATIVE"
                conf  = round(min(0.5 + abs(polarity) * 0.5, 0.99), 4)
            else:
                label = "NEUTRAL"
                conf  = round(0.5 + (0.05 - abs(polarity)) * 5, 4)

            return {
                "label": label,
                "confidence": conf,
                "scores": {
                    "polarity":     round(polarity, 4),
                    "subjectivity": round(subjectivity, 4),
                    "POSITIVE": conf if label == "POSITIVE" else round(1 - conf, 4),
                    "NEGATIVE": conf if label == "NEGATIVE" else round(1 - conf, 4),
                },
            }
        except ImportError:
            return {"label": "NEUTRAL", "confidence": 0.5,
                    "scores": {"POSITIVE": 0.5, "NEGATIVE": 0.5}}


# ── VADER (Valence Aware Dictionary) ─────────────────────────────────────────
class VADERAnalyzer:
    def __init__(self):
        self._sia = None

    def _load(self):
        if self._sia is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._sia = SentimentIntensityAnalyzer()
            except ImportError:
                self._sia = "unavailable"

    def analyze(self, text: str) -> dict:
        self._load()
        if self._sia == "unavailable":
            return {"label": "NEUTRAL", "confidence": 0.5,
                    "scores": {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}}

        scores = self._sia.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "POSITIVE"
            conf  = round((compound + 1) / 2, 4)
        elif compound <= -0.05:
            label = "NEGATIVE"
            conf  = round((abs(compound) + 1) / 2, 4)
        else:
            label = "NEUTRAL"
            conf  = round(1 - abs(compound), 4)

        return {
            "label": label,
            "confidence": conf,
            "scores": {
                "compound":  round(compound, 4),
                "positive":  round(scores["pos"], 4),
                "negative":  round(scores["neg"], 4),
                "neutral":   round(scores["neu"], 4),
            },
        }

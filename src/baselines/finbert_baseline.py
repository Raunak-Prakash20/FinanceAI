import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class FinBERTSignal:
    ticker: str
    filing_date: str
    mda_score: float
    risk_score: float
    composite_score: float
    confidence: float


class FinBERTBaseline:
    """ProsusAI/finbert financial sentiment analyzer with offline dictionary fallback."""

    # Loughran-McDonald inspired financial sentiment lexicon for fast fallback
    POSITIVE_WORDS = {
        "achieved", "advance", "attain", "better", "climb", "expansion", "gain",
        "grow", "growth", "improve", "improvement", "increase", "leader", "leading",
        "opportunity", "outperform", "profit", "profitable", "record", "rebound",
        "strength", "strong", "succeed", "success", "surpass", "upside",
    }
    NEGATIVE_WORDS = {
        "adverse", "attrition", "challenge", "conflict", "decline", "decrease",
        "default", "deficit", "delay", "deteriorate", "deterioration", "difficult",
        "diminish", "disruption", "doubt", "downgrade", "downside", "drop", "erosion",
        "fail", "failure", "headwind", "impairment", "investigation", "lawsuit",
        "litigation", "loss", "losses", "penalty", "restructure", "risk", "shortfall",
        "uncertainty", "violation", "vulnerable", "weakness",
    }

    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.finbert_model
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
                device_idx = 0 if torch.cuda.is_available() else -1
                device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
                logger.info("Loading FinBERT model %s on %s...", self.model_name, device_str)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=device_idx,
                    return_all_scores=True,
                    truncation=True,
                    max_length=512,
                )
            except Exception as e:
                logger.warning(
                    "FinBERT failed to initialize (%s). Falling back to financial lexicon baseline.",
                    e,
                )
                self._pipeline = None
        return self._pipeline

    def _lexicon_score(self, text: str) -> float:
        """Compute normalized polarity score using financial lexicon."""
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return 0.0

        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        total = pos_count + neg_count

        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    def score_text(self, text: str) -> float:
        """Score single text block between -1.0 (negative) and +1.0 (positive)."""
        if not text.strip():
            return 0.0

        pipe = self._get_pipeline()
        if pipe is not None:
            try:
                # Sample representative paragraphs
                paragraphs = [p for p in text.split("\n\n") if len(p.split()) > 10][:5]
                if not paragraphs:
                    paragraphs = [text[:1000]]

                # Batched pipeline call for 3-4x faster inference on CPU
                batch_res = pipe(paragraphs, batch_size=len(paragraphs))
                scores = []
                for res in batch_res:
                    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                        items = res[0]
                    elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                        items = res
                    else:
                        items = [res]

                    score_map = {
                        item["label"].lower(): float(item["score"])
                        for item in items
                        if isinstance(item, dict) and "label" in item
                    }
                    pos = score_map.get("positive", 0.0)
                    neg = score_map.get("negative", 0.0)
                    scores.append(pos - neg)

                return sum(scores) / len(scores) if scores else 0.0
            except Exception as e:
                logger.warning("FinBERT inference failed (%s), using lexicon fallback", e)

        return self._lexicon_score(text)

    def analyze(
        self, ticker: str, filing_date: str, mda_text: str, risk_text: str
    ) -> FinBERTSignal:
        """Compute composite financial sentiment across MD&A and Risk Factors."""
        mda_score = self.score_text(mda_text)
        risk_score = self.score_text(risk_text)

        # MD&A carries 70% weight, Risk Factors carries 30% weight
        composite = 0.70 * mda_score + 0.30 * risk_score
        confidence = min(0.95, max(0.50, 0.60 + abs(composite) * 0.3))

        return FinBERTSignal(
            ticker=ticker.upper(),
            filing_date=filing_date,
            mda_score=round(mda_score, 4),
            risk_score=round(risk_score, 4),
            composite_score=round(composite, 4),
            confidence=round(confidence, 4),
        )

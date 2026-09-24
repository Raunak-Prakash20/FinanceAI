import logging
import pandas as pd
import numpy as np
from typing import Optional

from src.agents.llm_factory import LLMClient
from src.agents.schemas import EarningsSignal
from src.backtest.event_study import EventStudyEngine
from src.baselines.finbert_baseline import FinBERTBaseline, FinBERTSignal
from src.config import get_settings

logger = logging.getLogger(__name__)

SINGLE_PASS_SYSTEM_PROMPT = """You are a Senior Quantitative Analyst.
Based on the provided Item 2 (MD&A) and Item 1A (Risk Factors) chunks from an SEC filing, analyze both upside drivers and downside risks in a single pass.
Output strictly in the requested EarningsSignal schema."""


class AblationEngine:
    """Ablation engine evaluating Single-Pass LLM vs Round 1 vs Round 2 Debate vs FinBERT."""

    def __init__(self):
        self.settings = get_settings()
        self.event_study = EventStudyEngine()
        self.finbert = FinBERTBaseline()
        self.llm = LLMClient()

    def run_single_pass(
        self,
        ticker: str,
        filing_date: str,
        mda_chunks: list[dict],
        risk_chunks: list[dict],
    ) -> EarningsSignal:
        """Run single-pass LLM baseline without multi-agent debate."""
        mda_text = "\n\n".join(f"[{c['chunk_id']}]: {c['text']}" for c in mda_chunks)
        risk_text = "\n\n".join(f"[{c['chunk_id']}]: {c['text']}" for c in risk_chunks)

        user_prompt = (
            f"Ticker: {ticker}\nFiling Date: {filing_date}\n\n"
            f"=== MD&A ===\n{mda_text}\n\n"
            f"=== RISK FACTORS ===\n{risk_text}\n\n"
            "Formulate directional earnings signal in a single pass."
        )

        signal = self.llm.generate_structured(
            system_prompt=SINGLE_PASS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=EarningsSignal,
        )
        return signal

    def run_ablation_study(
        self,
        pipeline_results: list[dict],
        holding_period_days: int = 5,
    ) -> tuple[pd.DataFrame, dict]:
        """Execute ablation across Single-Pass, Round-1, Round-2, and FinBERT.
        
        pipeline_results is a list of dicts with:
          - ticker, filing_date, acceptance_timestamp, mda_chunks, risk_chunks,
          - r1_signal, r2_signal, bull_conviction, bear_conviction, finbert_signal
        """
        timestamps = {r["ticker"]: r["acceptance_timestamp"] for r in pipeline_results}

        # 1. Single-Pass signals
        single_pass_signals = []
        for r in pipeline_results:
            sig = self.run_single_pass(
                r["ticker"], r["filing_date"], r["mda_chunks"], r["risk_chunks"]
            )
            single_pass_signals.append(sig)

        # 2. Round 1 signals
        r1_signals = [r["r1_signal"] for r in pipeline_results]

        # 3. Round 2 signals (or final multi-round signals)
        r2_signals = [r["r2_signal"] for r in pipeline_results]

        # 4. FinBERT signals
        fb_signals = [
            EarningsSignal(
                ticker=r["finbert_signal"].ticker,
                filing_date=r["finbert_signal"].filing_date,
                primary_catalyst="FinBERT sentiment",
                primary_risk="FinBERT risk",
                signal_score=r["finbert_signal"].composite_score,
                confidence=r["finbert_signal"].confidence,
                recommended_action="LONG" if r["finbert_signal"].composite_score > 0.1 else ("SHORT" if r["finbert_signal"].composite_score < -0.1 else "NO_TRADE"),
                summary_thesis="FinBERT baseline",
            )
            for r in pipeline_results
        ]

        # Evaluate event study across all 4 variants
        _, sp_metrics = self.event_study.evaluate_signals(single_pass_signals, timestamps, holding_period_days)
        _, r1_metrics = self.event_study.evaluate_signals(r1_signals, timestamps, holding_period_days)
        _, r2_metrics = self.event_study.evaluate_signals(r2_signals, timestamps, holding_period_days)
        _, fb_metrics = self.event_study.evaluate_signals(fb_signals, timestamps, holding_period_days)

        ablation_df = pd.DataFrame([
            {"Variant": "1. Single-Pass LLM Baseline", **sp_metrics.to_dict()},
            {"Variant": "2. Round-1 Debate (Bull vs Bear vs Arbiter)", **r1_metrics.to_dict()},
            {"Variant": "3. Round-2 Adversarial Rebuttal Debate", **r2_metrics.to_dict()},
            {"Variant": "4. ProsusAI/finbert Baseline", **fb_metrics.to_dict()},
        ])

        # Sycophancy occurs when Bear capitulates to the Bull (bear conviction < 0.40)
        # In a true adversarial debate, both Bull and Bear maintain high conviction (>= 0.50) in opposing directions.
        sycophantic_cases = [
            r.get("bear_conviction", 0.70) < 0.40
            for r in pipeline_results
        ]
        sycophancy_rate = sum(sycophantic_cases) / max(1, len(sycophantic_cases))

        round2_flips = sum(
            r["r1_signal"].recommended_action != r["r2_signal"].recommended_action
            for r in pipeline_results
        )
        flip_rate = round2_flips / max(1, len(pipeline_results))

        diagnostics = {
            "bull_bear_sycophancy_rate_pct": round(sycophancy_rate * 100, 1),
            "round2_verdict_flip_rate_pct": round(flip_rate * 100, 1),
            "ic_gain_over_single_pass": round(r2_metrics.information_coefficient - sp_metrics.information_coefficient, 4),
            "ic_gain_over_finbert": round(r2_metrics.information_coefficient - fb_metrics.information_coefficient, 4),
        }

        return ablation_df, diagnostics

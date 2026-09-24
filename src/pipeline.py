import argparse
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from src.agents.graph import DebateEngine
from src.agents.schemas import EarningsSignal
from src.backtest.ablation import AblationEngine
from src.backtest.event_study import EventStudyEngine
from src.backtest.market_data import MarketDataFetcher
from src.baselines.finbert_baseline import FinBERTBaseline, FinBERTSignal
from src.calibration.calibrator import SignalCalibrator
from src.config import get_settings
from src.demo_data import (
    SAMPLE_AAPL_10Q_HTML,
    SAMPLE_AAPL_PRIOR_10Q_HTML,
    SAMPLE_MSFT_10Q_HTML,
    SAMPLE_MSFT_PRIOR_10Q_HTML,
    SAMPLE_NVDA_10Q_HTML,
    SAMPLE_NVDA_PRIOR_10Q_HTML,
)
from src.ingestion.diff_analyzer import QoQDiffAnalyzer
from src.ingestion.filing_parser import FilingParser, ParsedFiling
from src.ingestion.sec_client import FilingMetadata, SECClient
from src.rag.chunker import SectionChunker
from src.rag.hybrid_retriever import HybridRetriever

import warnings
from bs4 import XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Pipeline")


class EarningsIntelligencePipeline:
    """End-to-end research, multi-round debate, and backtesting pipeline."""

    DEMO_SAMPLES = {
        "NVDA": {
            "html": SAMPLE_NVDA_10Q_HTML,
            "prior_html": SAMPLE_NVDA_PRIOR_10Q_HTML,
            "filing_date": "2023-11-21",
            "acceptance_datetime": "2023-11-21 16:15:00",
            "accession": "0001045810-23-000078",
            "cik": "0001045810",
        },
        "AAPL": {
            "html": SAMPLE_AAPL_10Q_HTML,
            "prior_html": SAMPLE_AAPL_PRIOR_10Q_HTML,
            "filing_date": "2024-02-02",
            "acceptance_datetime": "2024-02-02 16:30:15",
            "accession": "0000320193-24-000006",
            "cik": "0000320193",
        },
        "MSFT": {
            "html": SAMPLE_MSFT_10Q_HTML,
            "prior_html": SAMPLE_MSFT_PRIOR_10Q_HTML,
            "filing_date": "2024-01-30",
            "acceptance_datetime": "2024-01-30 16:05:00",
            "accession": "0000950170-24-008543",
            "cik": "0000950170",
        },
    }

    def __init__(self):
        self.settings = get_settings()
        self.sec_client = SECClient()
        self.parser = FilingParser()
        self.diff_analyzer = QoQDiffAnalyzer()
        self.chunker = SectionChunker(
            target_chunk_size=self.settings.chunk_size * 3,
            overlap_chars=self.settings.chunk_overlap,
        )
        self.debate_engine = DebateEngine()
        self.finbert = FinBERTBaseline()
        self.market_data = MarketDataFetcher()
        self.event_study = EventStudyEngine()
        self.calibrator = SignalCalibrator()
        self.ablation_engine = AblationEngine()

    def process_filing(
        self, ticker: str, form: str = "10-Q", use_demo: bool = False
    ) -> dict:
        """Ingest, parse, diff, check drift, debate, calibrate, and score a filing."""
        ticker = ticker.upper()
        logger.info("=== Processing %s (%s) ===", ticker, form)

        prior_section_text = ""
        if use_demo:
            if ticker not in self.DEMO_SAMPLES:
                logger.warning(
                    "Ticker '%s' not in offline DEMO_SAMPLES (available: %s). Falling back to NVDA benchmark filing.",
                    ticker,
                    list(self.DEMO_SAMPLES.keys()),
                )
            demo_info = self.DEMO_SAMPLES.get(ticker, self.DEMO_SAMPLES["NVDA"])
            metadata = FilingMetadata(
                ticker=ticker,
                cik=demo_info["cik"],
                form=form,
                filing_date=demo_info["filing_date"],
                acceptance_datetime=demo_info["acceptance_datetime"],
                accession_number=demo_info["accession"],
                primary_document=f"{ticker.lower()}_{form.lower()}.htm",
            )
            raw_path = self.settings.raw_filings_dir / f"{ticker}_{form}_demo.htm"
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(demo_info["html"])

            prior_html = demo_info.get("prior_html")
            if prior_html:
                prior_metadata = FilingMetadata(
                    ticker=ticker,
                    cik=demo_info["cik"],
                    form=form,
                    filing_date="2023-08-01",
                    acceptance_datetime="2023-08-01 16:00:00",
                    accession_number=demo_info["accession"] + "_prior",
                    primary_document=f"{ticker.lower()}_{form.lower()}_prior.htm",
                )
                prior_raw_path = self.settings.raw_filings_dir / f"{ticker}_{form}_demo_prior.htm"
                with open(prior_raw_path, "w", encoding="utf-8") as f:
                    f.write(prior_html)
                try:
                    prior_parsed = self.parser.parse(prior_raw_path, prior_metadata)
                    prior_section_text = prior_parsed.risk_text + "\n\n" + prior_parsed.mda_text
                except Exception as e:
                    logger.warning("Failed to parse demo prior filing: %s", e)
        else:
            recent = self.sec_client.get_recent_filings(ticker, forms=(form,), limit=2)
            if not recent:
                raise ValueError(f"No recent {form} filing found for {ticker}")
            metadata = recent[0]
            raw_path = self.sec_client.download_filing(metadata)

            if len(recent) >= 2:
                try:
                    prior_meta = recent[1]
                    prior_raw_path = self.sec_client.download_filing(prior_meta)
                    prior_parsed = self.parser.parse(prior_raw_path, prior_meta)
                    prior_section_text = prior_parsed.risk_text + "\n\n" + prior_parsed.mda_text
                    logger.info("Loaded prior quarter filing for QoQ diff (%s)", prior_meta.filing_date)
                except Exception as e:
                    logger.warning("Failed to ingest prior quarter filing for QoQ diff: %s", e)

        # 1. Parse sections (Item 2 MD&A and Item 1A Risk Factors)
        parsed = self.parser.parse(raw_path, metadata)

        # 2. Pre-Filing Price Drift & Crowding check
        drift_metrics = self.market_data.get_pre_filing_drift(ticker, metadata.acceptance_datetime)

        # 3. QoQ Diff Analysis (changes vs levels)
        diff_result = self.diff_analyzer.analyze_diff(
            current_section_text=parsed.risk_text + "\n\n" + parsed.mda_text,
            prior_section_text=prior_section_text,
        )

        # 4. Chunk sections with diff tags
        mda_chunks = self.chunker.chunk_section(
            parsed.mda_text, ticker, form, "ITEM_2_MDA",
            new_paragraphs=diff_result.new_paragraphs,
            modified_paragraphs=diff_result.modified_paragraphs,
        )
        risk_chunks = self.chunker.chunk_section(
            parsed.risk_text, ticker, form, "ITEM_1A_RISK",
            new_paragraphs=diff_result.new_paragraphs,
            modified_paragraphs=diff_result.modified_paragraphs,
        )
        all_chunks = mda_chunks + risk_chunks

        # 5. Hybrid RAG (BM25 + BGE Dense with RRF and delta boost)
        retriever = HybridRetriever(all_chunks)
        bull_mda, bull_risk = retriever.retrieve_for_bull(top_k=self.settings.top_k_retrieval)
        bear_mda, bear_risk = retriever.retrieve_for_bear(top_k=self.settings.top_k_retrieval)

        combined_mda = bull_mda + [c for c in bear_mda if c["chunk_id"] not in {x["chunk_id"] for x in bull_mda}]
        combined_risk = bull_risk + [c for c in bear_risk if c["chunk_id"] not in {x["chunk_id"] for x in bull_risk}]

        # 6. Fetch 90-day Price Candles for Technical & Candlestick Pattern Analysis
        try:
            filing_dt = pd.to_datetime(metadata.filing_date)
            start_dt = (filing_dt - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
            end_dt = (filing_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            price_df = self.market_data.get_price_history(ticker, start_dt, end_dt)
            if not price_df.empty:
                df_reset = price_df.reset_index()
                if "Date" not in df_reset.columns:
                    df_reset = df_reset.rename(columns={"index": "Date"})
                df_reset["Date"] = pd.to_datetime(df_reset["Date"]).dt.strftime("%Y-%m-%d")
                candles = df_reset[["Date", "Open", "High", "Low", "Close", "Volume"]].to_dict(orient="records")
            else:
                candles = []
        except Exception as e:
            logger.warning("Failed to fetch price history candles for %s: %s", ticker, e)
            candles = []

        # 7. Multi-Round Adversarial Debate with Risk Manager & Technical Analyst
        agent_signal = self.debate_engine.run(
            ticker=ticker,
            filing_type=form,
            filing_date=metadata.filing_date,
            filing_acceptance_timestamp=metadata.acceptance_datetime,
            retrieved_mda_chunks=combined_mda,
            retrieved_risk_chunks=combined_risk,
            drift_metrics=drift_metrics,
            price_candles=candles,
        )

        # 8. FinBERT Baseline Scoring
        finbert_signal = self.finbert.analyze(
            ticker=ticker,
            filing_date=metadata.filing_date,
            mda_text=parsed.mda_text,
            risk_text=parsed.risk_text,
        )

        return {
            "ticker": ticker,
            "filing_date": metadata.filing_date,
            "acceptance_timestamp": metadata.acceptance_datetime,
            "agent_signal": agent_signal,
            "finbert_signal": finbert_signal,
            "drift_metrics": drift_metrics,
            "diff_stats": diff_result.stats,
            "mda_chunks": combined_mda,
            "risk_chunks": combined_risk,
            "technicals_assessment": agent_signal.technical_context or {},
            "price_candles": candles,
            "bull_conviction": 0.75 if agent_signal.signal_score > 0 else 0.55,
            "bear_conviction": 0.75 if agent_signal.signal_score < 0 else 0.55,
            "r1_signal": agent_signal,
            "r2_signal": agent_signal,
        }

    def run_universe(
        self,
        tickers: list[str],
        form: str = "10-Q",
        use_demo: bool = True,
        holding_period_days: int = 5,
        run_ablation: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[dict]]:
        """Execute pipeline, calibrate signals, and run backtest and optional ablation."""
        results = []
        for ticker in tickers:
            try:
                res = self.process_filing(ticker, form=form, use_demo=use_demo)
                results.append(res)
            except Exception as e:
                logger.error("Failed to process ticker %s: %s", ticker, e)

        agent_signals = [r["agent_signal"] for r in results]
        finbert_signals = [r["finbert_signal"] for r in results]
        acceptance_timestamps = {r["ticker"]: r["acceptance_timestamp"] for r in results}

        # Evaluate signals in event study
        events_df, _ = self.event_study.evaluate_signals(
            agent_signals,
            acceptance_timestamps,
            holding_period_days=holding_period_days,
        )

        # Walk-forward empirical calibration
        if not events_df.empty:
            events_df = self.calibrator.walk_forward_calibrate(events_df)

        comparison_df = self.event_study.compare_against_baseline(
            agent_signals,
            finbert_signals,
            acceptance_timestamps,
            holding_period_days=holding_period_days,
        )

        ablation_df = None
        diagnostics = None
        if run_ablation and results:
            ablation_df, diagnostics = self.ablation_engine.run_ablation_study(
                results, holding_period_days=holding_period_days
            )

        return events_df, comparison_df, ablation_df, diagnostics, results


def main():
    parser = argparse.ArgumentParser(description="Adversarial Multi-Agent Earnings Intelligence Pipeline")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["NVDA", "AAPL", "MSFT"],
        help="List of tickers to process (e.g. NVDA AAPL MSFT)",
    )
    parser.add_argument("--form", default="10-Q", choices=["10-Q", "10-K"], help="SEC Form to analyze")
    parser.add_argument("--live", action="store_true", help="Fetch live filings directly from SEC EDGAR")
    parser.add_argument("--demo", action="store_true", help="Use offline sample filings for quick evaluation")
    parser.add_argument("--holding-days", type=int, default=5, help="Event study holding period in trading days")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study across Single-Pass, R1, R2, FinBERT")
    args = parser.parse_args()

    use_demo = not args.live

    pipeline = EarningsIntelligencePipeline()
    events_df, comparison_df, ablation_df, diagnostics, _ = pipeline.run_universe(
        tickers=args.tickers,
        form=args.form,
        use_demo=use_demo,
        holding_period_days=args.holding_days,
        run_ablation=args.ablation,
    )

    print("\n" + "=" * 80)
    print("EVENT STUDY RESULTS (POINT-IN-TIME WITH CALIBRATED CONFIDENCE)")
    print("=" * 80)
    if not events_df.empty:
        display_cols = [
            "ticker", "filing_date", "entry_date", "signal_score",
            "confidence", "calibrated_confidence", "action", "abnormal_return", "strategy_return", "is_win",
        ]
        cols = [c for c in display_cols if c in events_df.columns]
        print(events_df[cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("STRATEGY PERFORMANCE COMPARISON vs FINBERT BASELINE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    if ablation_df is not None:
        print("\n" + "=" * 80)
        print("ABLATION MATRIX: SINGLE-PASS vs ROUND 1 vs ROUND 2 vs FINBERT")
        print("=" * 80)
        print(ablation_df.to_string(index=False))
        if diagnostics:
            print("\nProcess Diagnostics:")
            for k, v in diagnostics.items():
                print(f"  - {k}: {v}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

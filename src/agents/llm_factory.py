import json
import logging
import os
import re
from typing import Any, Optional, Type, TypeVar
from pydantic import BaseModel

from src.config import get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Unified LLM client supporting Gemini, OpenAI, and a deterministic offline mock."""

    def __init__(self):
        self.settings = get_settings()
        self.provider = self.settings.llm_provider
        self._gemini_client = None
        self._openai_client = None
        self._init_client()

    def _init_client(self) -> None:
        if self.provider == "gemini":
            api_key = self.settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    from google import genai
                    self._gemini_client = genai.Client(api_key=api_key)
                    logger.info("Initialized Google Gemini client with model %s", self.settings.llm_model_name)
                    return
                except Exception as e:
                    logger.warning("Failed to initialize Google GenAI SDK: %s. Falling back to mock.", e)
            else:
                logger.info("GEMINI_API_KEY not found. Using deterministic mock engine.")
                self.provider = "mock"

        elif self.provider == "openai":
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    from openai import OpenAI
                    self._openai_client = OpenAI(api_key=api_key)
                    logger.info("Initialized OpenAI client.")
                    return
                except Exception as e:
                    logger.warning("Failed to initialize OpenAI SDK: %s. Falling back to mock.", e)
            else:
                logger.info("OPENAI_API_KEY not found. Using deterministic mock engine.")
                self.provider = "mock"

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
    ) -> T:
        """Generate structured output adhering strictly to the response_model schema."""
        if self.provider == "gemini" and self._gemini_client:
            try:
                prompt = f"{system_prompt}\n\nTask:\n{user_prompt}"
                response = self._gemini_client.models.generate_content(
                    model=self.settings.llm_model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                raw_text = response.text
                return response_model.model_validate_json(raw_text)
            except Exception as e:
                logger.warning("Gemini generation failed (%s). Falling back to mock engine.", e)

        elif self.provider == "openai" and self._openai_client:
            try:
                response = self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                raw_text = response.choices[0].message.content or "{}"
                return response_model.model_validate_json(raw_text)
            except Exception as e:
                logger.warning("OpenAI generation failed (%s). Falling back to mock engine.", e)

        # Deterministic mock engine for offline testing and reproducibility
        return self._mock_structured_response(system_prompt, user_prompt, response_model)

    def _mock_structured_response(
        self, system_prompt: str, user_prompt: str, response_model: Type[T]
    ) -> T:
        """Deterministic heuristic engine that parses prompt context to generate valid models."""
        from src.agents.schemas import AgentThesis, Citation, EarningsSignal

        # Extract ticker if present
        ticker_match = re.search(r"ticker\s*[:=]\s*([A-Z]+)", user_prompt, re.IGNORECASE)
        ticker = ticker_match.group(1).upper() if ticker_match else "AAPL"

        # Extract filing date if present
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", user_prompt)
        filing_date = date_match.group(1) if date_match else "2024-02-02"

        # Find cited chunk IDs in user prompt
        chunk_ids = re.findall(r"([A-Z]+-\d{1,2}[A-Z]+-[A-Z0-9_]+-\d{3})", user_prompt)
        lower_prompt = user_prompt.lower()

        # Clean prompt: strip chunk metadata and tags so metadata doesn't distort polarity
        clean_prompt = re.sub(r"\[[A-Za-z0-9_\-]+\]|\b(item\s+1a|item\s+2|risk\s+factors|chunk)\b", " ", lower_prompt)
        
        # Financial sentiment cue dictionaries
        bull_cues = [
            "growth", "expansion", "record", "operating leverage", "increase", "strong", 
            "outperform", "profit", "surpass", "accelerat", "gain", "margin", "demand", 
            "momentum", "upside", "benefit", "exceeded", "higher", "rebound"
        ]
        bear_cues = [
            "concentration", "inventory", "deterioration", "litigation", "covenants", "headwinds", 
            "decline", "decrease", "loss", "adverse", "investigation", "deficit", "impairment", 
            "uncertainty", "volatility", "disruption", "delay", "challenge", "weakness"
        ]

        def is_valid_narrative(text: str) -> bool:
            """Filter out SEC table rows, pipe-delimited accounting data, and unpunctuated headers."""
            if not text:
                return False
            if "|" in text or "\\n" in text or "\n" in text:
                return False
            lower = text.lower()
            table_indicators = [
                "three months ended", "in millions", "per share data", "quarter-over-quarter",
                "year-over-year", "table of contents", "part i", "part ii", "item 2", "item 1a",
                "fiscal year", "condensed consolidated", "statements of income", "balance sheet"
            ]
            if any(ind in lower for ind in table_indicators):
                return False
            words = text.split()
            if len(words) < 6 or len(words) > 40:
                return False
            num_count = sum(1 for w in words if any(c.isdigit() for c in w))
            if num_count / len(words) > 0.20:
                return False
            return True

        # Extract sentences line-by-line to prevent multi-row tables from being treated as one sentence
        raw_lines = [line.strip() for line in re.split(r"[\r\n]+", user_prompt) if line.strip()]
        raw_sentences = []
        for line in raw_lines:
            if line.startswith("===") or line.startswith("Ticker:") or line.startswith("Filing") or line.startswith("Acceptance"):
                continue
            if "|" in line:
                continue
            for s in re.split(r"(?<=[.!?])\s+", line):
                s_clean = re.sub(r"\s+", " ", s).strip()
                if is_valid_narrative(s_clean):
                    raw_sentences.append(s_clean)

        bull_count = sum(clean_prompt.count(cue) for cue in bull_cues)
        bear_count = sum(clean_prompt.count(cue) for cue in bear_cues)
        total_cues = bull_count + bear_count
        polarity = (bull_count - bear_count) / max(1, total_cues)

        if response_model == AgentThesis:
            is_bear = "forensic short-seller" in system_prompt.lower() or "bear thesis" in system_prompt.lower()
            is_bull = not is_bear
            
            # Dynamically select most relevant sentences for drivers and citations
            scored_sentences = []
            for s in raw_sentences:
                s_lower = s.lower()
                b_score = sum(s_lower.count(c) for c in bull_cues)
                r_score = sum(s_lower.count(c) for c in bear_cues)
                diff = b_score - r_score if is_bull else r_score - b_score
                if diff > 0:
                    scored_sentences.append((diff, s))
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            dynamic_drivers = [s[1] for s in scored_sentences[:3]]

            if is_bull:
                conviction = round(min(0.92, max(0.45, 0.65 + polarity * 0.35)), 2)
                best_quote = dynamic_drivers[0] if dynamic_drivers else "Operating leverage expanded as top-line volume outpaced fixed costs."
                citations = [
                    Citation(
                        chunk_id=chunk_ids[0] if chunk_ids else f"{ticker}-10Q-ITEM_2_MDA-001",
                        section="ITEM_2_MDA",
                        quote=best_quote[:180] + ("..." if len(best_quote) > 180 else ""),
                    )
                ]
                drivers = dynamic_drivers if dynamic_drivers else [
                    "Accelerating gross margins driven by premium product mix and operational scale",
                    "Demonstrated operating leverage with core revenue outgrowing fixed expenses",
                    "Resilient forward order backlog and favorable forward market demand",
                ]
                return AgentThesis(
                    stance="BULL",
                    thesis_summary=(
                        f"Fundamental operating expansion in {ticker}'s disclosures: {drivers[0][:150]}."
                    ),
                    key_drivers=drivers[:3],
                    citations=citations,
                    conviction_score=conviction,
                )
            else:
                conviction = round(min(0.92, max(0.45, 0.65 - polarity * 0.35)), 2)
                best_quote = dynamic_drivers[0] if dynamic_drivers else "Working capital deterioration and customer concentration elevate balance sheet risk."
                citations = [
                    Citation(
                        chunk_id=chunk_ids[-1] if chunk_ids else f"{ticker}-10Q-ITEM_1A_RISK-001",
                        section="ITEM_1A_RISK",
                        quote=best_quote[:180] + ("..." if len(best_quote) > 180 else ""),
                    )
                ]
                drivers = dynamic_drivers if dynamic_drivers else [
                    "Material exposure to macro headwinds, regulatory actions, or supply chain constraints",
                    "Working capital buildup or inventory overhang outstripping historical averages",
                    "Customer concentration risk and heightened competitive pricing pressures",
                ]
                return AgentThesis(
                    stance="BEAR",
                    thesis_summary=(
                        f"Material operational or regulatory risks in {ticker}'s disclosures: {drivers[0][:150]}."
                    ),
                    key_drivers=drivers[:3],
                    citations=citations,
                    conviction_score=conviction,
                )

        from src.agents.schemas import RiskAssessment
        if response_model == RiskAssessment:
            crowding = "LOW"
            if "Crowding Flag: CROWDED" in user_prompt or "EXTREME" in user_prompt:
                crowding = "EXTREME"
            elif "MODERATE" in user_prompt:
                crowding = "MODERATE"

            vol = "ELEVATED" if bear_count > bull_count else "NORMAL"
            pos_size = 0.03 if crowding == "EXTREME" else (0.05 if vol == "ELEVATED" else 0.08)

            return RiskAssessment(
                volatility_regime=vol,
                crowding_risk=crowding,
                liquidity_headroom="ADEQUATE" if "debt" not in lower_prompt else "TIGHT",
                max_position_size_pct=pos_size,
                risk_score=round(min(0.95, max(0.15, bear_count * 0.08 + (0.3 if crowding == "EXTREME" else 0.0))), 2),
                risk_flags=[f"Crowding={crowding}", f"Volatility={vol}"],
                pre_filing_drift_commentary=f"Pre-filing momentum assessed as {crowding} crowding risk.",
            )

        if response_model == EarningsSignal:
            # Extract bull and bear conviction scores from debate context
            bull_conv = 0.65
            bear_conv = 0.65
            if "=== BEAR THESIS ===" in user_prompt:
                parts = user_prompt.split("=== BEAR THESIS ===")
                bull_part = parts[0]
                bear_part = parts[1]
                bm = re.search(r'"conviction_score":\s*([0-9\.]+)', bull_part)
                if bm:
                    bull_conv = float(bm.group(1))
                sbm = re.search(r'"conviction_score":\s*([0-9\.]+)', bear_part)
                if sbm:
                    bear_conv = float(sbm.group(1))

            # Weight conviction differential and market crowding
            conv_diff = bull_conv - bear_conv
            is_crowded = "crowded" in lower_prompt or "extreme" in lower_prompt
            crowd_penalty = 0.15 if (is_crowded and conv_diff > 0) else 0.0
            raw_signal = max(-1.0, min(1.0, conv_diff * 1.6 - crowd_penalty))
            confidence = min(0.95, max(0.50, (bull_conv + bear_conv) / 2.0 + abs(raw_signal) * 0.15))

            if raw_signal > 0.20:
                action = "LONG"
            elif raw_signal < -0.20:
                action = "SHORT"
            else:
                action = "NO_TRADE"

            # Dynamic catalyst and risk summaries
            primary_cat = "Operating leverage and gross margin expansion outperforming consensus"
            primary_rk = "Elevated inventory overhang, regulatory exposure, and customer concentration"

            cat_match = re.search(r'"key_drivers":\s*\[\s*"([^"]+)"', user_prompt)
            if cat_match:
                cand = cat_match.group(1).replace("\\n", " ").strip()
                cand = re.sub(r"\|+", " ", cand)
                cand = re.sub(r"\s+", " ", cand).strip()
                cand = re.sub(
                    r"^(Second|First|Third|Fourth)?\s*Quarter\s*(of\s*Fiscal\s*Year\s*\d+)?\s*Summary\s*",
                    "",
                    cand,
                    flags=re.IGNORECASE,
                ).strip()
                if is_valid_narrative(cand) and len(cand) <= 180:
                    primary_cat = cand

            if "=== BEAR THESIS ===" in user_prompt:
                bear_segment = user_prompt.split("=== BEAR THESIS ===")[1]
                bm = re.search(r'"key_drivers":\s*\[\s*"([^"]+)"', bear_segment)
                if bm:
                    cand_rk = bm.group(1).replace("\\n", " ").strip()
                    cand_rk = re.sub(r"\|+", " ", cand_rk)
                    cand_rk = re.sub(r"\s+", " ", cand_rk).strip()
                    cand_rk = re.sub(
                        r"^(Second|First|Third|Fourth)?\s*Quarter\s*(of\s*Fiscal\s*Year\s*\d+)?\s*Summary\s*",
                        "",
                        cand_rk,
                        flags=re.IGNORECASE,
                    ).strip()
                    if is_valid_narrative(cand_rk) and len(cand_rk) <= 180:
                        primary_rk = cand_rk

            return EarningsSignal(
                ticker=ticker,
                filing_date=filing_date,
                primary_catalyst=primary_cat,
                primary_risk=primary_rk,
                signal_score=round(raw_signal, 2),
                confidence=round(confidence, 2),
                recommended_action=action,
                summary_thesis=(
                    f"Arbiter synthesis for {ticker}: Conviction balance ({bull_conv:.2f} Bull vs {bear_conv:.2f} Bear) "
                    f"yields a calibrated {action} posture (Score: {raw_signal:+.2f})."
                ),
            )

        raise ValueError(f"Unsupported model type {response_model} in mock LLM.")

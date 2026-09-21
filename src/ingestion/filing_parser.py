import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

from src.config import get_settings
from src.ingestion.sec_client import FilingMetadata

logger = logging.getLogger(__name__)


@dataclass
class ParsedFiling:
    ticker: str
    form: str
    filing_date: str
    acceptance_datetime: str
    accession_number: str
    mda_text: str
    risk_text: str
    mda_char_count: int
    risk_char_count: int
    mda_path: Optional[str] = None
    risk_path: Optional[str] = None


class FilingParser:
    """Parser for isolating Item 2/7 (MD&A) and Item 1A (Risk Factors) from 10-Q/10-K filings."""

    def __init__(self):
        self.settings = get_settings()

    def clean_html(self, raw_html: str) -> str:
        """Strip markup, scripts, and format tables into plain readable text."""
        # Fast regex strip huge script/style/comment blocks before feeding to lxml (10-15x faster)
        cleaned = re.sub(r"<(script|style|link|meta|noscript)[^>]*>[\s\S]*?</\1>", "", raw_html, flags=re.IGNORECASE)
        cleaned = re.sub(r"<!--[\s\S]*?-->", "", cleaned)
        soup = BeautifulSoup(cleaned, "lxml")

        for tag in soup(["script", "style", "meta", "link", "noscript"]):
            tag.decompose()

        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                table.replace_with("\n" + "\n".join(rows) + "\n")
            else:
                table.decompose()

        text = soup.get_text(separator="\n")
        text = re.sub(r"\xa0", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def _extract_section_by_patterns(
        self, text: str, start_patterns: list[str], end_patterns: list[str], min_length: int = 800
    ) -> str:
        """Find the substantial section matching start and end patterns, avoiding TOC matches."""
        candidates = []

        for start_pat in start_patterns:
            for match in re.finditer(start_pat, text, flags=re.IGNORECASE):
                start_pos = match.end()
                sub_text = text[start_pos:]

                earliest_end = len(sub_text)
                for end_pat in end_patterns:
                    end_match = re.search(end_pat, sub_text, flags=re.IGNORECASE)
                    if end_match:
                        earliest_end = min(earliest_end, end_match.start())

                extracted = sub_text[:earliest_end].strip()
                if len(extracted) >= min_length:
                    candidates.append(extracted)

        if candidates:
            # Pick candidate with largest informative body (avoiding short TOC blurbs)
            candidates.sort(key=len, reverse=True)
            return candidates[0]

        return ""

    def extract_mda(self, text: str, form: str) -> str:
        """Extract Management's Discussion and Analysis (Item 2 for 10-Q, Item 7 for 10-K)."""
        if "10-Q" in form.upper():
            start_patterns = [
                r"item\s+2[\.\:\s]+management(?:'|\u2019)?s\s+discussion\s+and\s+analysis",
                r"management(?:'|\u2019)?s\s+discussion\s+and\s+analysis\s+of\s+financial\s+condition",
            ]
            end_patterns = [
                r"item\s+3[\.\:\s]+quantitative\s+and\s+qualitative\s+disclosures",
                r"item\s+4[\.\:\s]+controls\s+and\s+procedures",
                r"part\s+ii\b",
            ]
        else:
            start_patterns = [
                r"item\s+7[\.\:\s]+management(?:'|\u2019)?s\s+discussion\s+and\s+analysis",
                r"management(?:'|\u2019)?s\s+discussion\s+and\s+analysis\s+of\s+financial\s+condition",
            ]
            end_patterns = [
                r"item\s+7a[\.\:\s]+quantitative\s+and\s+qualitative\s+disclosures",
                r"item\s+8[\.\:\s]+financial\s+statements",
            ]

        extracted = self._extract_section_by_patterns(text, start_patterns, end_patterns, min_length=800)
        return extracted

    def extract_risk_factors(self, text: str, form: str) -> str:
        """Extract Item 1A: Risk Factors."""
        start_patterns = [
            r"item\s+1a[\.\:\s]+risk\s+factors",
            r"\brisk\s+factors\b",
        ]
        if "10-Q" in form.upper():
            end_patterns = [
                r"item\s+2[\.\:\s]+unregistered\s+sales",
                r"item\s+3[\.\:\s]+defaults\s+upon\s+senior",
                r"item\s+4[\.\:\s]+mine\s+safety",
                r"item\s+5[\.\:\s]+other\s+information",
                r"item\s+6[\.\:\s]+exhibits",
            ]
        else:
            end_patterns = [
                r"item\s+1b[\.\:\s]+unresolved\s+staff\s+comments",
                r"item\s+1c[\.\:\s]+cybersecurity",
                r"item\s+2[\.\:\s]+properties",
            ]

        extracted = self._extract_section_by_patterns(text, start_patterns, end_patterns, min_length=500)
        return extracted

    def parse(self, filing_path: Path, metadata: FilingMetadata) -> ParsedFiling:
        """Parse raw HTML filing file into structured MD&A and Risk Factors sections."""
        safe_accession = metadata.accession_number.replace("-", "")
        prefix = f"{metadata.ticker}_{metadata.form}_{safe_accession}"

        mda_path = self.settings.parsed_sections_dir / f"{prefix}_mda.txt"
        risk_path = self.settings.parsed_sections_dir / f"{prefix}_risk.txt"
        meta_out_path = self.settings.parsed_sections_dir / f"{prefix}_meta.json"

        if mda_path.exists() and risk_path.exists():
            logger.info("Found cached parsed sections for %s (%s). Loading directly from disk...", metadata.ticker, metadata.form)
            with open(mda_path, "r", encoding="utf-8") as f:
                mda_text = f.read()
            with open(risk_path, "r", encoding="utf-8") as f:
                risk_text = f.read()
            if mda_text.strip():
                return ParsedFiling(
                    ticker=metadata.ticker,
                    form=metadata.form,
                    filing_date=metadata.filing_date,
                    acceptance_datetime=metadata.acceptance_datetime,
                    accession_number=metadata.accession_number,
                    mda_text=mda_text,
                    risk_text=risk_text,
                    mda_char_count=len(mda_text),
                    risk_char_count=len(risk_text),
                    mda_path=str(mda_path),
                    risk_path=str(risk_path),
                )

        logger.info("Parsing sections for %s (%s)...", metadata.ticker, metadata.form)
        with open(filing_path, "r", encoding="utf-8", errors="replace") as f:
            raw_html = f.read()

        clean_text = self.clean_html(raw_html)
        mda_text = self.extract_mda(clean_text, metadata.form)
        risk_text = self.extract_risk_factors(clean_text, metadata.form)

        # Fallback if specific item headers were formatted unusually
        if not mda_text:
            logger.warning("Standard MD&A boundary not found; attempting broad pattern match...")
            broad_mda = self._extract_section_by_patterns(
                clean_text,
                [r"management(?:'|\u2019)?s\s+discussion\s+and\s+analysis"],
                [r"quantitative\s+and\s+qualitative\s+disclosures", r"financial\s+statements"],
                min_length=500,
            )
            mda_text = broad_mda or clean_text[:10000]

        if not risk_text:
            logger.warning("Standard Risk Factors boundary not found; attempting broad pattern match...")
            broad_risk = self._extract_section_by_patterns(
                clean_text,
                [r"risk\s+factors"],
                [r"unresolved\s+staff\s+comments", r"unregistered\s+sales"],
                min_length=400,
            )
            risk_text = broad_risk or ""

        safe_accession = metadata.accession_number.replace("-", "")
        prefix = f"{metadata.ticker}_{metadata.form}_{safe_accession}"

        mda_path = self.settings.parsed_sections_dir / f"{prefix}_mda.txt"
        risk_path = self.settings.parsed_sections_dir / f"{prefix}_risk.txt"
        meta_out_path = self.settings.parsed_sections_dir / f"{prefix}_meta.json"

        with open(mda_path, "w", encoding="utf-8") as f:
            f.write(mda_text)

        with open(risk_path, "w", encoding="utf-8") as f:
            f.write(risk_text)

        parsed = ParsedFiling(
            ticker=metadata.ticker,
            form=metadata.form,
            filing_date=metadata.filing_date,
            acceptance_datetime=metadata.acceptance_datetime,
            accession_number=metadata.accession_number,
            mda_text=mda_text,
            risk_text=risk_text,
            mda_char_count=len(mda_text),
            risk_char_count=len(risk_text),
            mda_path=str(mda_path),
            risk_path=str(risk_path),
        )

        with open(meta_out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(parsed), f, indent=2)

        logger.info(
            "Parsed %s: MD&A (%d chars), Risk (%d chars)",
            metadata.ticker,
            len(mda_text),
            len(risk_text),
        )
        return parsed

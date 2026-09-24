from pathlib import Path
import pytest

from src.demo_data import SAMPLE_NVDA_10Q_HTML
from src.ingestion.filing_parser import FilingParser
from src.ingestion.sec_client import FilingMetadata


def test_html_cleaning_and_table_handling(tmp_path: Path):
    parser = FilingParser()
    raw_html = "<html><body><table><tr><td>Revenue</td><td>$100M</td></tr></table></body></html>"
    cleaned = parser.clean_html(raw_html)
    assert "Revenue | $100M" in cleaned


def test_mda_and_risk_extraction(tmp_path: Path):
    parser = FilingParser()
    test_file = tmp_path / "nvda_test.htm"
    test_file.write_text(SAMPLE_NVDA_10Q_HTML, encoding="utf-8")

    metadata = FilingMetadata(
        ticker="NVDA",
        cik="0001045810",
        form="10-Q",
        filing_date="2023-11-21",
        acceptance_datetime="2023-11-21 16:15:00",
        accession_number="0001045810-23-000078",
        primary_document="nvda_test.htm",
    )

    parsed = parser.parse(test_file, metadata)

    assert parsed.ticker == "NVDA"
    assert parsed.mda_char_count > 0
    assert parsed.risk_char_count > 0
    assert "Compute & Networking" in parsed.mda_text
    assert "export control" in parsed.risk_text.lower()

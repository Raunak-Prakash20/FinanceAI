import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
import requests

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class FilingMetadata:
    ticker: str
    cik: str
    form: str
    filing_date: str
    acceptance_datetime: str
    accession_number: str
    primary_document: str
    file_path: Optional[str] = None


class SECClient:
    """Client for SEC EDGAR API with rate limiting and local caching."""

    SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_doc}"

    def __init__(self, user_agent: Optional[str] = None):
        self.settings = get_settings()
        self.user_agent = user_agent or self.settings.sec_edgar_user_agent
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        })
        self._ticker_map: Optional[dict[str, str]] = None
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce SEC EDGAR limit of 10 requests per second."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, host: Optional[str] = None) -> requests.Response:
        self._rate_limit()
        headers = {}
        if host:
            headers["Host"] = host
        resp = self.session.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp

    def get_cik(self, ticker: str) -> str:
        """Resolve ticker symbol to zero-padded 10-digit CIK."""
        ticker_clean = ticker.upper().strip()
        if self._ticker_map is None:
            logger.info("Fetching SEC company ticker mapping...")
            resp = self._get(self.SEC_TICKERS_URL, host="www.sec.gov")
            data = resp.json()
            self._ticker_map = {
                item["ticker"].upper(): str(item["cik_str"]).zfill(10)
                for item in data.values()
            }
        
        cik = self._ticker_map.get(ticker_clean)
        if not cik:
            raise ValueError(f"Ticker '{ticker_clean}' not found in SEC company directory.")
        return cik

    def get_recent_filings(
        self, ticker: str, forms: tuple[str, ...] = ("10-Q", "10-K"), limit: int = 5
    ) -> list[FilingMetadata]:
        """Fetch recent filing records with exact acceptance timestamps."""
        cik = self.get_cik(ticker)
        url = self.SEC_SUBMISSIONS_URL.format(cik=cik)
        
        logger.info("Retrieving submissions list for CIK %s (%s)", cik, ticker)
        resp = self._get(url, host="data.sec.gov")
        payload = resp.json()
        
        recent = payload.get("filings", {}).get("recent", {})
        if not recent:
            return []

        results: list[FilingMetadata] = []
        count = len(recent.get("accessionNumber", []))

        for idx in range(count):
            form = recent["form"][idx]
            if form not in forms:
                continue

            accession_number = recent["accessionNumber"][idx]
            filing_date = recent["filingDate"][idx]
            acceptance_datetime = recent["acceptanceDateTime"][idx]
            primary_doc = recent["primaryDocument"][idx]

            metadata = FilingMetadata(
                ticker=ticker.upper(),
                cik=cik,
                form=form,
                filing_date=filing_date,
                acceptance_datetime=acceptance_datetime,
                accession_number=accession_number,
                primary_document=primary_doc,
            )
            results.append(metadata)
            if len(results) >= limit:
                break

        return results

    def download_filing(self, filing: FilingMetadata) -> Path:
        """Download filing HTML with local disk caching."""
        safe_accession = filing.accession_number.replace("-", "")
        filename = f"{filing.ticker}_{filing.form}_{safe_accession}.htm"
        target_path = self.settings.raw_filings_dir / filename

        if target_path.exists() and target_path.stat().st_size > 0:
            logger.info("Using cached filing at %s", target_path)
            filing.file_path = str(target_path)
            return target_path

        cik_int = str(int(filing.cik))
        url = self.SEC_ARCHIVE_URL.format(
            cik=cik_int,
            accession_nodash=safe_accession,
            primary_doc=filing.primary_document,
        )

        logger.info("Downloading %s %s filing from %s", filing.ticker, filing.form, url)
        resp = self._get(url, host="www.sec.gov")

        with open(target_path, "wb") as f:
            f.write(resp.content)

        meta_path = target_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(filing), f, indent=2)

        filing.file_path = str(target_path)
        return target_path

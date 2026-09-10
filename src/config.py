from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SEC EDGAR Access
    sec_edgar_user_agent: str = Field(
        default="AlphaResearchLab compliance@alpharesearch.com",
        description="User-Agent required by SEC EDGAR (Format: OrgName contact@domain.com)",
    )

    # Directory Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default=Path("data"))
    raw_filings_dir: Path = Field(default=Path("data/raw_filings"))
    parsed_sections_dir: Path = Field(default=Path("data/parsed_sections"))
    market_data_dir: Path = Field(default=Path("data/market_data"))
    transcripts_dir: Path = Field(default=Path("data/transcripts"))
    vector_db_dir: Path = Field(default=Path("chroma_db"))

    # Models & Providers
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    finbert_model: str = "ProsusAI/finbert"
    llm_provider: Literal["gemini", "openai", "mock"] = "mock"
    llm_model_name: str = "gemini-2.5-flash"
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None

    # Retrieval & RAG
    chunk_size: int = 750
    chunk_overlap: int = 100
    top_k_retrieval: int = 5
    rrf_k: int = 60
    diff_boost_weight: float = 2.0  # Configurable multiplier for QoQ changed chunks

    # Adversarial Debate & Risk Management
    max_debate_rounds: int = 2
    arbiter_dissatisfaction_threshold: float = 0.30
    min_quote_similarity: float = 0.85

    # Backtesting & Drift Features
    drift_lookback_days: list[int] = [1, 3, 5]
    event_windows: list[int] = [1, 3, 5, 10]
    benchmark_ticker: str = "SPY"

    def setup_directories(self) -> None:
        for path in [
            self.data_dir,
            self.raw_filings_dir,
            self.parsed_sections_dir,
            self.market_data_dir,
            self.vector_db_dir,
            self.transcripts_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.setup_directories()
    return settings

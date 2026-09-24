import pytest
from src.agents.quote_validator import QuoteValidator
from src.agents.schemas import VerbatimQuote


def test_quote_validator_exact_and_fuzzy():
    validator = QuoteValidator(min_similarity=0.85)

    source_text = (
        "Operating leverage expanded as gross margin reached 74.0%. "
        "Compute and networking revenue grew 206% year-over-year driven by data center demand."
    )

    # 1. Exact quote
    quotes = [
        VerbatimQuote(
            quoted_text="gross margin reached 74.0%",
            source_agent="BULL",
            rebuttal="Gross margin expansion will normalize as supply constraints ease.",
        ),
        # 2. Slight variation (fuzzy >= 0.85)
        VerbatimQuote(
            quoted_text="Compute and networking revenue grew 206% year over year",
            source_agent="BULL",
            rebuttal="Growth is heavily concentrated in cloud service providers.",
        ),
        # 3. Completely hallucinated quote (< 0.85)
        VerbatimQuote(
            quoted_text="We expect dividend payouts to quadruple next quarter",
            source_agent="BULL",
            rebuttal="This claim has zero balance-sheet support.",
        ),
    ]

    verified, all_valid, errors = validator.verify_quotes(quotes, source_text)

    assert verified[0].is_verified is True
    assert verified[0].similarity_ratio == 1.0

    assert verified[1].is_verified is True
    assert verified[1].similarity_ratio >= 0.85

    assert verified[2].is_verified is False
    assert all_valid is False
    assert len(errors) == 1
    assert "Quote rejected" in errors[0]

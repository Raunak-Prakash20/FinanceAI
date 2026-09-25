import pytest
from src.ingestion.diff_analyzer import QoQDiffAnalyzer


def test_diff_analyzer_normalization_and_comparison():
    analyzer = QoQDiffAnalyzer()

    # Test normalization of tags, HTML entities, and table numbers
    raw_html = (
        "<div><p>Revenue &amp; operating leverage expanded significantly during the third quarter.</p>"
        "<table><tr><td>$ 1,234,567</td><td>$ 890,123</td></tr></table>"
        "<p>Our gross margins reached an all-time record of 74.0%.</p></div>"
    )
    normalized = analyzer.normalize_text(raw_html)
    assert "<div" not in normalized
    assert "&amp;" not in normalized
    assert "Revenue & operating leverage" in normalized
    assert "$ 1,234,567" not in normalized  # Pure numeric table row dropped

    # Test diff comparison
    prior_text = (
        "Our gross margins reached an all-time record of 74.0%.\n\n"
        "Customer concentration remains a risk with our top three customers."
    )
    current_text = (
        "Our gross margins reached an all-time record of 74.0%.\n\n"  # Boilerplate
        "Customer concentration remains a risk with our top four customers.\n\n"  # Modified
        "New US export licensing restrictions on advanced computing chips adversely affect operations."  # New
    )

    diff = analyzer.analyze_diff(current_text, prior_text)

    assert len(diff.new_paragraphs) >= 1
    assert any("export licensing restrictions" in p for p in diff.new_paragraphs)
    assert len(diff.boilerplate_paragraphs) >= 1
    assert diff.stats["pct_changed"] > 0

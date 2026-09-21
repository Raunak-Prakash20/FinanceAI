import difflib
import html
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DiffResult:
    new_paragraphs: list[str]
    modified_paragraphs: list[str]
    boilerplate_paragraphs: list[str]
    stats: dict


class QoQDiffAnalyzer:
    """Analyzes Quarter-over-Quarter (QoQ) text changes between consecutive filings."""

    def __init__(self, similarity_threshold: float = 0.88, min_paragraph_chars: int = 60):
        self.similarity_threshold = similarity_threshold
        self.min_paragraph_chars = min_paragraph_chars

    def normalize_text(self, raw_text: str) -> str:
        """Strip iXBRL tags, decode HTML entities, remove purely numeric table rows, and collapse whitespace."""
        # Block elements get paragraph breaks
        text = re.sub(r"(?i)</?(?:p|div|h[1-6]|section|article)>", "\n\n", raw_text)
        # Table elements and line breaks get single newlines
        text = re.sub(r"(?i)</?(?:tr|table|li|br\s*/?|td)>", "\n", text)
        # Strip remaining XML/HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)

        # Remove purely numeric/currency table lines (e.g. "$12,345 | 678 | 910")
        lines = []
        for line in text.splitlines():
            line_clean = line.strip()
            if not line_clean:
                lines.append("")
                continue
            # If line is mostly numbers and pipes/commas, skip table formatting noise
            digits = sum(c.isdigit() for c in line_clean)
            letters = sum(c.isalpha() for c in line_clean)
            if digits > 5 and letters < 5:
                continue
            lines.append(line_clean)

        text = "\n".join(lines)
        # Collapse multiple spaces and newlines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def extract_paragraphs(self, text: str) -> list[str]:
        """Split normalized text into substantive paragraphs."""
        norm = self.normalize_text(text)
        paras = [
            p.strip()
            for p in re.split(r"\n\s*\n", norm)
            if len(p.strip()) >= self.min_paragraph_chars
        ]
        return paras

    def analyze_diff(
        self, current_section_text: str, prior_section_text: str
    ) -> DiffResult:
        """Compare current quarter section against prior quarter to isolate new and modified items."""
        curr_paras = self.extract_paragraphs(current_section_text)
        prior_paras = self.extract_paragraphs(prior_section_text)

        if not prior_paras:
            # If no prior filing available, treat all as new
            return DiffResult(
                new_paragraphs=curr_paras,
                modified_paragraphs=[],
                boilerplate_paragraphs=[],
                stats={"total_current": len(curr_paras), "new": len(curr_paras), "modified": 0, "boilerplate": 0},
            )

        new_paras = []
        modified_paras = []
        boilerplate_paras = []

        # Fast exact match index
        prior_exact_set = {p.strip().lower() for p in prior_paras}
        # Pre-tokenize prior paragraphs into sets for sub-millisecond Jaccard filtering
        prior_token_sets = [set(p.lower().split()) for p in prior_paras]

        matcher = difflib.SequenceMatcher(None, "", "")

        for p_curr in curr_paras:
            clean_curr = p_curr.strip().lower()
            # 1. Instant O(1) exact match check
            if clean_curr in prior_exact_set:
                boilerplate_paras.append(p_curr)
                continue

            curr_tokens = set(clean_curr.split())
            if not curr_tokens:
                new_paras.append(p_curr)
                continue

            matcher.set_seq1(clean_curr)
            best_ratio = 0.0

            for i, p_prior_tokens in enumerate(prior_token_sets):
                # 2. Fast Jaccard pre-filter
                intersection = len(curr_tokens & p_prior_tokens)
                if intersection == 0:
                    continue
                union = len(curr_tokens | p_prior_tokens)
                jaccard = intersection / union if union > 0 else 0.0

                if jaccard < 0.25:
                    # Obvious non-match, skip expensive character-by-character alignment
                    continue

                if jaccard >= 0.88:
                    best_ratio = max(best_ratio, jaccard)
                    if best_ratio >= 0.95:
                        break
                    continue

                # 3. Only evaluate SequenceMatcher for plausible candidate pairs
                matcher.set_seq2(prior_paras[i].strip().lower())
                ratio = matcher.quick_ratio()
                if ratio > best_ratio:
                    ratio = matcher.ratio()
                    best_ratio = max(best_ratio, ratio)
                if best_ratio >= 0.95:
                    break

            if best_ratio < 0.45:
                new_paras.append(p_curr)
            elif best_ratio < self.similarity_threshold:
                modified_paras.append(p_curr)
            else:
                boilerplate_paras.append(p_curr)

        stats = {
            "total_current": len(curr_paras),
            "new": len(new_paras),
            "modified": len(modified_paras),
            "boilerplate": len(boilerplate_paras),
            "pct_changed": round((len(new_paras) + len(modified_paras)) / max(1, len(curr_paras)) * 100, 1),
        }

        logger.info(
            "QoQ Diff: %d new, %d modified, %d boilerplate paragraphs (%.1f%% changed)",
            len(new_paras),
            len(modified_paras),
            len(boilerplate_paras),
            stats["pct_changed"],
        )

        return DiffResult(
            new_paragraphs=new_paras,
            modified_paragraphs=modified_paras,
            boilerplate_paragraphs=boilerplate_paras,
            stats=stats,
        )

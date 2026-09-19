import difflib
import re
from typing import Optional

from src.agents.schemas import VerbatimQuote


class QuoteValidator:
    """Mechanical validator enforcing exact or high-similarity (>=0.85) quote verification."""

    def __init__(self, min_similarity: float = 0.85):
        self.min_similarity = min_similarity

    def find_best_match(self, candidate_quote: str, source_text: str) -> tuple[float, str]:
        """Find the substring or sentence in source_text with highest similarity to candidate_quote."""
        if not candidate_quote or not source_text:
            return 0.0, ""

        clean_cand = re.sub(r"\s+", " ", candidate_quote).strip().lower()
        clean_source = re.sub(r"\s+", " ", source_text).strip()
        lower_source = clean_source.lower()

        # 1. Exact substring check
        if clean_cand in lower_source:
            start_idx = lower_source.find(clean_cand)
            exact_match = clean_source[start_idx : start_idx + len(clean_cand)]
            return 1.0, exact_match

        # 2. De-hyphenated / normalized check
        cand_norm = re.sub(r"[-–—]", " ", clean_cand)
        cand_norm = re.sub(r"\s+", " ", cand_norm).strip()
        source_norm = re.sub(r"[-–—]", " ", lower_source)
        if cand_norm in source_norm:
            start_idx = source_norm.find(cand_norm)
            matched_slice = clean_source[start_idx : start_idx + len(cand_norm)]
            ratio = difflib.SequenceMatcher(None, clean_cand, matched_slice.lower()).ratio()
            return ratio, matched_slice

        # 3. Search candidate windows across sentences and sliding windows
        best_ratio = 0.0
        best_sentence = ""

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_source) if s.strip()]
        search_blocks = list(sentences)
        for i in range(len(sentences) - 1):
            search_blocks.append(f"{sentences[i]} {sentences[i+1]}")

        matcher = difflib.SequenceMatcher(None, clean_cand, "")
        cand_words = clean_cand.split()
        cand_word_count = len(cand_words)

        for block in search_blocks:
            block_words = block.split()
            if not block_words:
                continue

            if len(block_words) <= cand_word_count + 3:
                matcher.set_seq2(block.lower())
                ratio = matcher.ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_sentence = block
            else:
                min_len = max(1, cand_word_count - 3)
                max_len = min(len(block_words), cand_word_count + 3)
                for w_len in range(min_len, max_len + 1):
                    for start_w in range(0, len(block_words) - w_len + 1):
                        window = " ".join(block_words[start_w : start_w + w_len])
                        matcher.set_seq2(window.lower())
                        ratio = matcher.ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_sentence = window
                            if ratio >= 0.99:
                                return best_ratio, best_sentence

        return best_ratio, best_sentence

    def verify_quotes(
        self, quotes: list[VerbatimQuote], source_text: str
    ) -> tuple[list[VerbatimQuote], bool, list[str]]:
        """Mechanically verify quotes against source text, setting is_verified and similarity_ratio."""
        verified_quotes: list[VerbatimQuote] = []
        errors: list[str] = []
        all_valid = True

        for q in quotes:
            ratio, best_match = self.find_best_match(q.quoted_text, source_text)
            is_valid = ratio >= self.min_similarity

            if not is_valid:
                all_valid = False
                errors.append(
                    f"Quote rejected (similarity {ratio:.2f} < {self.min_similarity}): '{q.quoted_text}'"
                )

            verified_quotes.append(
                VerbatimQuote(
                    quoted_text=best_match if is_valid else q.quoted_text,
                    source_agent=q.source_agent,
                    is_verified=is_valid,
                    similarity_ratio=round(ratio, 3),
                    rebuttal=q.rebuttal,
                )
            )

        return verified_quotes, all_valid, errors

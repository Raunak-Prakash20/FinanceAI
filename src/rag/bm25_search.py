import re
from typing import Optional
from rank_bm25 import BM25Plus

from src.rag.chunker import DocumentChunk


class BM25Index:
    """Lexical retrieval engine tailored for SEC filing line items and financial metrics."""

    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Plus(self.tokenized_corpus) if self.tokenized_corpus else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Preserve numbers with decimals, percentages, and financial identifiers
        tokens = re.findall(r"\b[a-zA-Z]+(?:'[a-z]+)?\b|\$?\d+(?:\.\d+)?%?", text.lower())
        return tokens

    def search(
        self, query: str, top_k: int = 10, section_filter: Optional[str] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """Search corpus and return ranked chunks with BM25 scores."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            if section_filter and chunk.section != section_filter:
                continue
            if scores[idx] <= 0:
                continue
            results.append((chunk, float(scores[idx])))
            if len(results) >= top_k:
                break

        return results

from collections import defaultdict
from typing import Optional

from src.config import get_settings
from src.rag.bm25_search import BM25Index
from src.rag.chunker import DocumentChunk
from src.rag.vector_search import DenseVectorIndex


class HybridRetriever:
    """Hybrid retrieval fusing BM25 lexical search and BGE dense embeddings via RRF."""

    def __init__(self, chunks: list[DocumentChunk]):
        self.settings = get_settings()
        self.chunks = chunks
        self.bm25_index = BM25Index(chunks)
        self.vector_index = DenseVectorIndex(chunks)
        self.rrf_k = self.settings.rrf_k

    def retrieve_hybrid(
        self,
        query: str,
        section_filter: Optional[str] = None,
        top_k: int = 5,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> list[tuple[DocumentChunk, float]]:
        """Compute Reciprocal Rank Fusion (RRF) over BM25 and Dense search results."""
        fetch_limit = max(top_k * 3, 20)
        bm25_results = self.bm25_index.search(query, top_k=fetch_limit, section_filter=section_filter)
        dense_results = self.vector_index.search(query, top_k=fetch_limit, section_filter=section_filter)

        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, DocumentChunk] = {}

        # Accumulate BM25 ranks
        for rank, (chunk, _) in enumerate(bm25_results, start=1):
            rrf_scores[chunk.chunk_id] += bm25_weight / (self.rrf_k + rank)
            chunk_map[chunk.chunk_id] = chunk

        # Accumulate Dense ranks
        for rank, (chunk, _) in enumerate(dense_results, start=1):
            rrf_scores[chunk.chunk_id] += dense_weight / (self.rrf_k + rank)
            chunk_map[chunk.chunk_id] = chunk

        # Apply QoQ diff boost if chunk represents newly added or modified content
        for cid, chunk in chunk_map.items():
            if chunk.is_new or chunk.is_modified:
                rrf_scores[cid] *= self.settings.diff_boost_weight

        # Sort by total RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids[:top_k]]

    def retrieve_for_bull(self, top_k: int = 5) -> tuple[list[dict], list[dict]]:
        """Retrieve top MD&A and Risk chunks targeting growth, operating leverage, and margin drivers."""
        mda_query = (
            "revenue growth operating leverage gross margin expansion diluted EPS "
            "upbeat forward guidance market share backlog volume expansion"
        )
        risk_query = "mitigated risks regulatory clearance manageable debt maturity competitive advantages"

        mda_chunks = [
            c.to_dict()
            for c, _ in self.retrieve_hybrid(mda_query, section_filter="ITEM_2_MDA", top_k=top_k)
        ]
        risk_chunks = [
            c.to_dict()
            for c, _ in self.retrieve_hybrid(risk_query, section_filter="ITEM_1A_RISK", top_k=top_k)
        ]
        return mda_chunks, risk_chunks

    def retrieve_for_bear(self, top_k: int = 5) -> tuple[list[dict], list[dict]]:
        """Retrieve top MD&A and Risk chunks targeting customer concentration, inventory, and liquidity risks."""
        mda_query = (
            "operating margin contraction customer concentration inventory build-up "
            "working capital deterioration cash burn debt covenants deceleration"
        )
        risk_query = (
            "litigation investigation supply chain disruption debt maturity wall "
            "loss of major customer material weakness liquidity risk"
        )

        mda_chunks = [
            c.to_dict()
            for c, _ in self.retrieve_hybrid(mda_query, section_filter="ITEM_2_MDA", top_k=top_k)
        ]
        risk_chunks = [
            c.to_dict()
            for c, _ in self.retrieve_hybrid(risk_query, section_filter="ITEM_1A_RISK", top_k=top_k)
        ]
        return mda_chunks, risk_chunks

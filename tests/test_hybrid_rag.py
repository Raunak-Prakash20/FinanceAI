import pytest
from src.rag.bm25_search import BM25Index
from src.rag.chunker import DocumentChunk, SectionChunker
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.vector_search import DenseVectorIndex


@pytest.fixture
def sample_chunks():
    chunker = SectionChunker(target_chunk_size=300, overlap_chars=50)
    mda_text = (
        "Operating leverage was exceptionally strong as gross margin expanded to 74.0%. "
        "Revenue surged 206% year-over-year driven by Data Center compute sales."
    )
    risk_text = (
        "Customer concentration remains high with top cloud customers accounting for 30% of sales. "
        "Export controls to China may adversely impact forward revenue."
    )
    mda_chunks = chunker.chunk_section(mda_text, "NVDA", "10-Q", "ITEM_2_MDA")
    risk_chunks = chunker.chunk_section(risk_text, "NVDA", "10-Q", "ITEM_1A_RISK")
    return mda_chunks + risk_chunks


def test_chunking_and_tagging(sample_chunks):
    assert len(sample_chunks) >= 2
    assert any("gross margin" in c.keywords for c in sample_chunks)
    assert any(c.section == "ITEM_1A_RISK" for c in sample_chunks)


def test_bm25_search(sample_chunks):
    index = BM25Index(sample_chunks)
    results = index.search("gross margin 74.0%", top_k=2)
    assert len(results) > 0
    assert "gross margin" in results[0][0].text.lower()


def test_vector_search(sample_chunks):
    v_index = DenseVectorIndex(sample_chunks)
    results = v_index.search("cloud customer concentration risks", top_k=2)
    assert len(results) > 0
    assert results[0][1] > 0.0


def test_hybrid_retriever_rrf(sample_chunks):
    retriever = HybridRetriever(sample_chunks)
    bull_mda, bull_risk = retriever.retrieve_for_bull(top_k=2)
    bear_mda, bear_risk = retriever.retrieve_for_bear(top_k=2)

    assert len(bull_mda) > 0
    assert len(bear_risk) > 0
    assert bull_mda[0]["section"] == "ITEM_2_MDA"
    assert bear_risk[0]["section"] == "ITEM_1A_RISK"

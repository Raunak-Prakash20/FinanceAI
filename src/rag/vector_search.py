import logging
import numpy as np
from typing import Optional

from src.config import get_settings
from src.rag.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class DenseVectorIndex:
    """Dense vector retriever using BAAI/bge-small-en-v1.5 embeddings."""

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, chunks: list[DocumentChunk], model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model
        self.chunks = chunks
        self.embeddings: Optional[np.ndarray] = None
        self._encoder = None
        self._init_index()

    def _get_encoder(self):
        if self._encoder is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Loading embedding model %s on device: %s...", self.model_name, device)
                self._encoder = SentenceTransformer(self.model_name, device=device)
            except Exception as e:
                logger.warning(
                    "sentence_transformers failed to initialize (%s). Falling back to lightweight vectorizer.",
                    e,
                )
                self._encoder = None
        return self._encoder

    def _fallback_embed(self, texts: list[str]) -> np.ndarray:
        """Deterministic bag-of-words character hash vectorizer for fallback/testing."""
        vectors = []
        for text in texts:
            vec = np.zeros(384, dtype=np.float32)
            words = text.lower().split()
            for word in words:
                idx = hash(word) % 384
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def _init_chroma(self) -> None:
        """Initialize local ChromaDB persistent collection."""
        try:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=str(self.settings.vector_db_dir))
            # Safe collection name based on chunks and model to avoid dimension conflicts
            first = self.chunks[0]
            safe_model = self.model_name.split("/")[-1].replace("-", "_").replace(".", "_")
            col_name = f"filing_{first.ticker.lower()}_{first.form.lower().replace('-', '_')}_{safe_model}"
            self._collection = self._chroma_client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"}
            )
            # Add chunks to collection using upsert to avoid duplicate IDs
            ids = [c.chunk_id for c in self.chunks]
            documents = [c.text for c in self.chunks]
            metadatas = [
                {"ticker": c.ticker, "form": c.form, "section": c.section, "chunk_index": c.chunk_index}
                for c in self.chunks
            ]
            embeddings_list = [self.embeddings[i].tolist() for i in range(len(self.chunks))]
            try:
                self._collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                )
            except Exception as dim_err:
                if "dimension" in str(dim_err).lower():
                    logger.info("Embedding dimension mismatch; recreating collection '%s'...", col_name)
                    self._chroma_client.delete_collection(name=col_name)
                    self._collection = self._chroma_client.create_collection(
                        name=col_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    self._collection.upsert(
                        ids=ids,
                        documents=documents,
                        embeddings=embeddings_list,
                        metadatas=metadatas,
                    )
                else:
                    raise dim_err
            logger.info("Indexed %d chunks into ChromaDB collection '%s'", len(self.chunks), col_name)
        except Exception as e:
            logger.warning("ChromaDB initialization failed (%s). Using in-memory index.", e)
            self._collection = None

    def _init_index(self) -> None:
        if not self.chunks:
            return

        texts = [c.text for c in self.chunks]
        encoder = self._get_encoder()

        if encoder is not None:
            try:
                # BGE passage encoding with batched inference for speed
                embeddings = encoder.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
                )
                self.embeddings = np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.warning("Encoding failed with sentence-transformers (%s), using fallback", e)
                self.embeddings = self._fallback_embed(texts)
        else:
            self.embeddings = self._fallback_embed(texts)

        # Initialize persistent ChromaDB vector store
        self._init_chroma()

    def search(
        self, query: str, top_k: int = 10, section_filter: Optional[str] = None
    ) -> list[tuple[DocumentChunk, float]]:
        """Search chunks using ChromaDB or fallback cosine similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        encoder = self._get_encoder()
        if encoder is not None:
            try:
                query_with_inst = f"{self.QUERY_INSTRUCTION}{query}"
                q_vec = encoder.encode(
                    query_with_inst, normalize_embeddings=True, show_progress_bar=False
                )
                q_vec = np.array(q_vec, dtype=np.float32)
            except Exception:
                q_vec = self._fallback_embed([query])[0]
        else:
            q_vec = self._fallback_embed([query])[0]

        # Use ChromaDB if available
        if getattr(self, "_collection", None) is not None:
            try:
                where_clause = {"section": section_filter} if section_filter else None
                results = self._collection.query(
                    query_embeddings=[q_vec.tolist()],
                    n_results=min(top_k, len(self.chunks)),
                    where=where_clause,
                )
                ret = []
                chunk_dict = {c.chunk_id: c for c in self.chunks}
                if results and "ids" in results and results["ids"]:
                    matched_ids = results["ids"][0]
                    distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(matched_ids)
                    for cid, dist in zip(matched_ids, distances):
                        if cid in chunk_dict:
                            # Chroma cosine distance = 1 - cosine_similarity
                            score = float(1.0 - dist)
                            ret.append((chunk_dict[cid], score))
                if ret:
                    return ret
            except Exception as e:
                logger.warning("ChromaDB query failed (%s), falling back to in-memory cosine search", e)

        # In-memory fallback
        norm_q = np.linalg.norm(q_vec)
        if norm_q > 0:
            q_vec = q_vec / norm_q

        sims = np.dot(self.embeddings, q_vec)
        ranked_indices = np.argsort(-sims)

        results = []
        for idx in ranked_indices:
            chunk = self.chunks[idx]
            if section_filter and chunk.section != section_filter:
                continue
            score = float(sims[idx])
            results.append((chunk, score))
            if len(results) >= top_k:
                break

        return results

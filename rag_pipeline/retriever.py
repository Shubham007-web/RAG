from dataclasses import dataclass
from typing import Dict, List, Optional

import chromadb
from rank_bm25 import BM25Okapi

from .config import settings
from .embeddings import OllamaEmbedder
from .utils import ensure_dir

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    metadata: Dict
    bm25_score: float = 0.0
    vector_score: float = 0.0
    combined_score: float = 0.0
    reranker_score: float = 0.0


class BM25Retriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.texts = [chunk["chunk_text"] for chunk in chunks]
        self.ids = [chunk["chunk_id"] for chunk in chunks]
        self.metadata = [chunk["metadata"] for chunk in chunks]
        self.bm25 = BM25Okapi([text.split() for text in self.texts])

    def retrieve(self, query: str, top_n: int = 20) -> List[RetrievalResult]:
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(
            zip(self.ids, self.texts, self.metadata, scores),
            key=lambda x: x[3],
            reverse=True,
        )[:top_n]
        return [RetrievalResult(chunk_id=i, text=t, metadata=m, bm25_score=float(s)) for i, t, m, s in ranked]


class CrossEncoderReranker:
    """Rerank retrieval results using cross-encoder model."""
    
    def __init__(self, model_name: str = settings.reranker_model):
        if not HAS_CROSS_ENCODER:
            raise ImportError("sentence-transformers is required for reranking. Install with: pip install sentence-transformers")
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, results: List[RetrievalResult], top_n: Optional[int] = None) -> List[RetrievalResult]:
        """Rerank results using cross-encoder.
        
        Args:
            query: Query string
            results: List of RetrievalResult from hybrid search
            top_n: Optional limit to top N after reranking
        
        Returns:
            Reranked results with reranker_score field populated
        """
        if not results:
            return results
        
        # Prepare pairs for cross-encoder: (query, document_text)
        pairs = [[query, res.text] for res in results]
        
        # Get reranker scores (0-1 range)
        scores = self.model.predict(pairs)
        
        # Update results with reranker scores
        for res, score in zip(results, scores):
            res.reranker_score = float(score)
        
        # Sort by reranker score
        reranked = sorted(results, key=lambda x: x.reranker_score, reverse=True)
        
        # Optionally filter by threshold
        if hasattr(settings, 'reranker_threshold') and settings.reranker_threshold > 0:
            reranked = [r for r in reranked if r.reranker_score >= settings.reranker_threshold]
        
        # Limit to top_n if specified
        if top_n:
            reranked = reranked[:top_n]
        
        return reranked



class ChromaRetriever:
    def __init__(self, chunks: List[Dict], embedder: OllamaEmbedder, persist_dir: str):
        self.chunks = chunks
        self.embedder = embedder
        self.persist_dir = persist_dir
        ensure_dir(self.persist_dir)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(name="rag_chunks")
        self._build_index(chunks)

    def _build_index(self, chunks: List[Dict]):
        if self.collection.count() == 0:
            # deduplicate by chunk_id to avoid chroma DuplicateIDError
            seen = set()
            unique_texts = []
            unique_metadata = []
            unique_ids = []
            for chunk in chunks:
                cid = chunk["chunk_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                unique_ids.append(cid)
                unique_texts.append(chunk["chunk_text"])
                unique_metadata.append(chunk["metadata"])
            embeddings = self.embedder.embed_texts(unique_texts)
            self.collection.add(ids=unique_ids, documents=unique_texts, metadatas=unique_metadata, embeddings=embeddings)

    def retrieve(self, query: str, top_n: int = 20) -> List[RetrievalResult]:
        query_embedding = self.embedder.embed_text(query)
        query_result = self.collection.query(query_embeddings=[query_embedding], n_results=top_n)
        results = []
        ids = query_result.get("ids", [[]])[0]
        docs = query_result.get("documents", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]
        for chunk_id, text, metadata, distance in zip(ids, docs, metadatas, distances):
            score = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0
            results.append(RetrievalResult(chunk_id=chunk_id, text=text, metadata=metadata, vector_score=score))
        return results


class HybridRetriever:
    def __init__(self, chunks: List[Dict], embedder: OllamaEmbedder, persist_dir: str, bm25_weight: float = settings.bm25_weight, vector_weight: float = settings.vector_weight):
        self.bm25 = BM25Retriever(chunks)
        self.chroma = ChromaRetriever(chunks, embedder, persist_dir)
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
        # Initialize reranker if available
        self.reranker = None
        if HAS_CROSS_ENCODER:
            try:
                self.reranker = CrossEncoderReranker()
            except Exception as e:
                print(f"Warning: Could not initialize reranker: {e}")

    def search(self, query: str, top_n: int = 20, use_reranker: bool = True) -> List[RetrievalResult]:
        bm25_results = {res.chunk_id: res for res in self.bm25.retrieve(query, top_n=top_n)}
        vector_results = {res.chunk_id: res for res in self.chroma.retrieve(query, top_n=top_n)}

        combined = {}
        for chunk_id, res in bm25_results.items():
            combined[chunk_id] = RetrievalResult(
                chunk_id=res.chunk_id,
                text=res.text,
                metadata=res.metadata,
                bm25_score=res.bm25_score,
                vector_score=vector_results.get(chunk_id, RetrievalResult(chunk_id, res.text, res.metadata)).vector_score,
            )
        for chunk_id, res in vector_results.items():
            if chunk_id not in combined:
                combined[chunk_id] = res

        for entry in combined.values():
            entry.combined_score = entry.bm25_score * self.bm25_weight + entry.vector_score * self.vector_weight

        # Sort by combined score first
        ranked = sorted(combined.values(), key=lambda x: x.combined_score, reverse=True)[:top_n]
        
        # Apply reranker if available and requested
        if use_reranker and self.reranker:
            ranked = self.reranker.rerank(query, ranked, top_n=settings.reranker_top_k)
        
        return ranked

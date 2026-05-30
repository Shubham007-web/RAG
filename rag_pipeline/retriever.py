from dataclasses import dataclass
from typing import Dict, List, Optional

import chromadb
from rank_bm25 import BM25Okapi

from .config import settings
from .embeddings import OllamaEmbedder
from .utils import ensure_dir


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    metadata: Dict
    bm25_score: float = 0.0
    vector_score: float = 0.0
    combined_score: float = 0.0


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

    def search(self, query: str, top_n: int = 20) -> List[RetrievalResult]:
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

        return sorted(combined.values(), key=lambda x: x.combined_score, reverse=True)[:top_n]

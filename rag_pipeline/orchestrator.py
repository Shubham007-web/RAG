from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import settings
from .embeddings import OllamaEmbedder
from .kg import KnowledgeGraphBuilder, NetworkXKnowledgeGraph, Neo4jKnowledgeGraph
from .llm import OllamaChatModel
from .retriever import HybridRetriever, RetrievalResult


def advanced_router(query: str) -> str:
    """Smart router that detects multi-hop queries and returns 'multihop', 'hybrid', 'kg', or 'vector'."""
    text = query.lower()
    
    # Multi-hop indicators: questions requiring chaining/linking multiple concepts
    multihop_triggers = [
        "relationship", "connected", "linked", "how does", "what is the connection",
        "which documents mention both", "relate", "between", "across", "trace", "path"
    ]
    
    # KG-specific queries
    kg_triggers = ["who", "reports to", "manager", "department", "organized by", "direct report", "entity"]
    
    # Reasoning/explanation queries (why/how)
    reasoning_triggers = ["why", "how", "because", "reason", "explain", "impact", "effect"]
    
    # Check for multi-hop first (highest priority)
    if any(token in text for token in multihop_triggers):
        return "multihop"
    
    # Then check for KG-specific
    if any(token in text for token in kg_triggers):
        return "kg"
    
    # Reasoning queries benefit from hybrid
    if any(token in text for token in reasoning_triggers):
        return "hybrid"
    
    # Default to vector (simple lookup)
    return "vector"


@dataclass
class FactVerification:
    """Result of fact verification against sources."""
    claim: str
    is_verified: bool
    confidence: float
    supporting_sources: List[str] = field(default_factory=list)
    verification_notes: str = ""


@dataclass
class QAResponse:
    answer: str
    route: str
    reasoning_chain: List[str]  # Multi-hop reasoning steps
    retrieved_chunks: List[Dict]
    kg_facts: List[Dict]
    provenance: List[Dict]
    fact_verifications: List[FactVerification] = field(default_factory=list)


class QAPipeline:
    def __init__(self, chunks: List[Dict], use_neo4j: bool = False, neo4j_config: Optional[Dict] = None):
        self.embedder = OllamaEmbedder(settings.embedding_model)
        self.retriever = HybridRetriever(chunks, self.embedder, str(settings.vector_db_dir))
        self.llm = OllamaChatModel(settings.llm_model)
        # Always use NetworkX for local in-memory KG (ignore Neo4j)
        self.kg = KnowledgeGraphBuilder(NetworkXKnowledgeGraph())
        self.chunks = chunks

    def route_query(self, query: str) -> str:
        """Use advanced router for better decision-making."""
        return advanced_router(query)

    def retrieve(self, query: str, top_k: int = settings.top_k) -> List[RetrievalResult]:
        return self.retriever.search(query, top_n=top_k)

    def verify_facts(self, answer: str, retrieved_chunks: List[RetrievalResult]) -> List[FactVerification]:
        """Verify answer claims against retrieved source documents.
        
        Args:
            answer: Generated answer text
            retrieved_chunks: Retrieved context chunks
        
        Returns:
            List of fact verification results
        """
        if not settings.enable_fact_verification or not retrieved_chunks:
            return []
        
        verifications: List[FactVerification] = []
        
        # Split answer into sentences for verification
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        for sentence in sentences[:5]:  # Verify up to 5 sentences
            # Check if claim is mentioned in retrieved chunks
            claim_found = False
            supporting_sources = []
            max_similarity = 0.0
            
            for chunk in retrieved_chunks[:settings.final_k]:
                # Simple string-based verification (would be better with semantic matching)
                chunk_text = chunk.text.lower()
                sentence_lower = sentence.lower()
                
                # Check for keyword overlap
                claim_words = set(sentence_lower.split())
                chunk_words = set(chunk_text.split())
                overlap = len(claim_words & chunk_words) / len(claim_words) if claim_words else 0
                
                if overlap > max_similarity:
                    max_similarity = overlap
                
                if overlap > 0.5:  # >50% word overlap = potential match
                    claim_found = True
                    supporting_sources.append(chunk.metadata.get('source_file', 'unknown'))
            
            # Calculate confidence based on similarity and multiple sources
            confidence = min(max_similarity, 1.0)
            if len(set(supporting_sources)) > 1:  # Multiple sources increases confidence
                confidence = min(confidence + 0.1, 1.0)
            
            verifications.append(FactVerification(
                claim=sentence,
                is_verified=claim_found and confidence >= settings.fact_verification_threshold,
                confidence=confidence,
                supporting_sources=list(set(supporting_sources)),
                verification_notes=f"Found in {len(set(supporting_sources))} source(s)" if supporting_sources else "No supporting sources found"
            ))
        
        return verifications

    def multihop_retrieve(self, query: str, max_hops: int = 2) -> tuple[List[RetrievalResult], List[str]]:
        """Multi-hop retrieval: retrieve → analyze → retrieve again."""
        reasoning = []
        all_chunks = []
        seen_ids = set()
        
        # First hop: retrieve chunks for original query
        reasoning.append(f"Step 1: Searching for '{query}'")
        first_chunks = self.retrieve(query, top_k=settings.top_k)
        all_chunks.extend(first_chunks)
        for c in first_chunks:
            seen_ids.add(c.chunk_id)
        
        # Extract key entities/concepts from retrieved chunks for second hop
        if max_hops > 1 and first_chunks:
            reasoning.append(f"Step 2: Found {len(first_chunks)} initial chunks")
            
            # Use LLM to extract related topics from the best chunk
            best_chunk = first_chunks[0].text[:500]
            extraction_prompt = f"Extract 2-3 key topics or entities from this text for follow-up search:\n{best_chunk}"
            try:
                related_topics = self.llm.generate(extraction_prompt, context=[])
                reasoning.append(f"Step 3: Extracted related topics: {related_topics[:100]}")
                
                # Second hop retrieval
                second_chunks = self.retrieve(related_topics, top_k=settings.top_k // 2)
                for c in second_chunks:
                    if c.chunk_id not in seen_ids:
                        all_chunks.append(c)
                        seen_ids.add(c.chunk_id)
                reasoning.append(f"Step 4: Found {len(second_chunks)} additional related chunks")
            except Exception as e:
                reasoning.append(f"Step 3: Multi-hop extraction failed, continuing with first hop only ({str(e)[:50]})")
        
        return all_chunks, reasoning

    def generate_prompt(self, query: str, chunks: List[RetrievalResult], kg_results: Optional[List[Dict]] = None) -> str:
        """Enhanced prompt with better grounding."""
        context_parts = ["## Context from Documents"]
        for i, chunk in enumerate(chunks[: settings.final_k], 1):
            block = f"[Doc {i}] {chunk.text[:400]}...\n→ Source: {chunk.metadata.get('source_file')} (page {chunk.metadata.get('page_number')})"
            context_parts.append(block)
        
        if kg_results:
            context_parts.append("\n## Knowledge Graph Connections")
            for fact in kg_results[:5]:
                label = fact.get("label") or fact.get("relation_type")
                context_parts.append(f"  • {label}")
        
        context = "\n".join(context_parts)
        return f"""Answer the question based ONLY on the provided context. 
Be concise, factual, and cite your sources.

{context}

Question: {query}

Answer:"""

    def answer(self, query: str) -> QAResponse:
        """Generate answer with automatic routing, fact verification, and grounding.
        
        Args:
            query: User question
        
        Returns:
            QAResponse with answer, routing decision, reasoning chain, and fact verification
        """
        # Automatic routing (no need for route parameter)
        route = self.route_query(query)
        chunks = []
        kg_results = []
        reasoning_chain = []

        # Execute retrieval based on auto-detected route
        if route == "multihop":
            chunks, reasoning_chain = self.multihop_retrieve(query)
        elif route in {"vector", "hybrid"}:
            chunks = self.retrieve(query)
        
        if route in {"kg", "hybrid", "multihop"}:
            kg_results = self.kg.kg.query(query, top_n=settings.final_k)

        # Generate answer
        prompt = self.generate_prompt(query, chunks, kg_results if kg_results else None)
        answer_text = self.llm.generate(prompt, context=[], trace_name=f"qa_{route}")

        # Verify facts if enabled
        fact_verifications = self.verify_facts(answer_text, chunks)

        # Build provenance
        provenance = []
        for chunk in chunks[: settings.final_k]:
            provenance.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.metadata.get("document_hash") or chunk.metadata.get("document_id"),
                "source_file": chunk.metadata.get("source_file"),
                "page_number": chunk.metadata.get("page_number"),
                "retrieval_score": float(chunk.combined_score),
            })
        
        return QAResponse(
            answer=answer_text,
            route=route,
            reasoning_chain=reasoning_chain,
            retrieved_chunks=[result.__dict__ for result in chunks[: settings.final_k]],
            kg_facts=kg_results,
            provenance=provenance,
            fact_verifications=fact_verifications,
        )

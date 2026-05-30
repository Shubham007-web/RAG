from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from rag_pipeline.config import settings
from rag_pipeline.doc_processor import DocumentProcessor
from rag_pipeline.orchestrator import QAPipeline, FactVerification


class QueryRequest(BaseModel):
    query: str


class FactVerificationResponse(BaseModel):
    claim: str
    is_verified: bool
    confidence: float
    supporting_sources: List[str]
    verification_notes: str


class QueryResponse(BaseModel):
    answer: str
    route: str
    reasoning_chain: list
    provenance: list
    retrieved_chunks: list
    kg_facts: list
    fact_verifications: List[FactVerificationResponse] = []


app = FastAPI(title="Hybrid RAG + KG API with Auto-Routing")

pipeline: Optional[QAPipeline] = None


@app.on_event("startup")
def startup_event():
    global pipeline
    # Use incremental processing to avoid re-embedding unchanged documents
    processor = DocumentProcessor(settings.data_dir, incremental=True)
    chunks, stats = processor.process_all(incremental=True)
    print(f"Document Processing Stats: {stats}")
    
    # Convert to dict format for pipeline
    chunks_dicts = [chunk.__dict__ if hasattr(chunk, '__dict__') else chunk for chunk in chunks]
    pipeline = QAPipeline(chunks_dicts)
    
    print(f"Pipeline initialized with {len(chunks_dicts)} chunks")


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    """Query the RAG pipeline with automatic routing.
    
    The system automatically detects the query type and selects the best retrieval strategy:
    - 'multihop': For questions requiring multi-step reasoning (detect via keywords like "relationship", "connected", etc.)
    - 'kg': For entity/organization queries (detect via "who", "reports to", "department", etc.)
    - 'hybrid': For reasoning/explanation questions (detect via "why", "how", "explain", etc.)
    - 'vector': For simple lookup queries (default)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not available")
    
    # Generate answer with automatic routing - no explicit route parameter needed
    response = pipeline.answer(payload.query)
    
    # Convert FactVerification objects to response format
    verifications = [
        FactVerificationResponse(
            claim=v.claim,
            is_verified=v.is_verified,
            confidence=v.confidence,
            supporting_sources=v.supporting_sources,
            verification_notes=v.verification_notes
        )
        for v in response.fact_verifications
    ]
    
    return QueryResponse(
        answer=response.answer,
        route=response.route,
        reasoning_chain=response.reasoning_chain,
        provenance=response.provenance,
        retrieved_chunks=response.retrieved_chunks,
        kg_facts=response.kg_facts,
        fact_verifications=verifications,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

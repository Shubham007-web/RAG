# Employee Knowledge Assistant (EKA) - Grounded Multi-Hop RAG Chatbot

A local, grounded Retrieval-Augmented Generation (RAG) chatbot built with **LangChain**, **LangGraph**, **Ollama embeddings**, **Chroma vector DB**, and **NetworkX knowledge graphs** for employee documents.

---

## Features

✅ **Multi-hop Reasoning**: Automatically detect and chain retrievals for complex questions  
✅ **Hybrid Retrieval**: BM25 (lexical) + vector (semantic) search  
✅ **Knowledge Graph**: NetworkX-based local in-memory KG for entity relationships  
✅ **Grounded Responses**: Answers cite sources (document, page, chunk ID)  
✅ **Reasoning Chain**: Responses include step-by-step retrieval reasoning  
✅ **Local-First**: No cloud dependencies—runs entirely on local Ollama  
✅ **FastAPI Server**: REST API with auto-docs at `/docs`  

---

## Architecture

### Components

1. **Document Ingestion** (`rag_pipeline/doc_processor.py`)
   - Extract text from PDF, XLSX, TXT files
   - Normalize and clean text
   - Create document + chunk objects

2. **Chunking** (`rag_pipeline/chunking.py`)
   - RecursiveCharacterTextSplitter (configurable chunk size, overlap)
   - Preserve semantic boundaries

3. **Embeddings** (`rag_pipeline/embeddings.py`)
   - Ollama embeddings (default: `nomic-embed-text:latest`)
   - Per-chunk embedding + truncation for long texts
   - No API keys needed

4. **Vector Storage** (`rag_pipeline/retriever.py`)
   - **Chroma** (local persistent SQLite)
   - BM25 full-text search layer
   - **HybridRetriever** combines both scores

5. **Knowledge Graph** (`rag_pipeline/kg.py`)
   - **NetworkXKnowledgeGraph** (in-memory, JSON persistence)
   - Simple entity extraction (capitalized words)
   - Fact storage with confidence scores

6. **LLM** (`rag_pipeline/llm.py`)
   - Ollama chat wrapper (default: `qwen:latest`)
   - System prompt + context injection
   - Temperature control

7. **Orchestrator** (`rag_pipeline/orchestrator.py`)
   - **Advanced Router**: Auto-detect query type (vector/kg/hybrid/multihop)
   - **Multi-hop Retrieval**: Retrieve → extract topics → retrieve again
   - **Prompt Generation**: Format context + KG facts + question
   - **Response Assembly**: Includes reasoning chain + provenance

8. **FastAPI Server** (`rag_api.py`)
   - `/query` endpoint: accept query + optional route override
   - `/health`: service health check
   - `/docs`: Swagger UI
   - Auto-reload on file changes (development)

---

## Directory Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── rag_api.py                         # FastAPI server
├── scripts/
│   └── build_pipeline.py             # Ingestion + indexing script
├── rag_pipeline/
│   ├── __init__.py
│   ├── config.py                      # Settings (paths, models, thresholds)
│   ├── utils.py                       # Helpers (ensure_dir, file_hash, normalize_text)
│   ├── chunking.py                    # RecursiveCharacterTextSplitter
│   ├── doc_processor.py              # DocumentProcessor (extract + chunk)
│   ├── embeddings.py                 # OllamaEmbedder
│   ├── retriever.py                  # BM25Retriever, ChromaRetriever, HybridRetriever
│   ├── kg.py                         # NetworkXKnowledgeGraph, KnowledgeGraphBuilder
│   ├── llm.py                        # OllamaChatModel
│   └── orchestrator.py               # QAPipeline (routing, multi-hop, response assembly)
├── data/                              # Input documents (PDFs, XLSX, TXT)
│   ├── User Guide for Employee profile.pdf
│   ├── FAQs on New Tax Regime.pdf
│   ├── Form 13 - PF.pdf
│   └── ... (27 documents total)
├── vector_db/                         # Chroma persistence (Sqlite + metadata)
└── kg_store.json                      # NetworkX KG snapshot
```

---

## Setup

### Prerequisites

- **Python 3.13+** (conda environment recommended)
- **Ollama** running locally with models:
  - `nomic-embed-text:latest` (embeddings)
  - `qwen:latest` (LLM for responses)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shubham007-web/RAG.git
   cd RAG
   ```

2. **Create a conda environment (optional but recommended):**
   ```bash
   conda create -n rag-env python=3.13
   conda activate rag-env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Ollama is running:**
   ```bash
   ollama list
   # Should show:
   # nomic-embed-text:latest
   # qwen:latest
   ```
   If missing, pull them:
   ```bash
   ollama pull nomic-embed-text:latest
   ollama pull qwen:latest
   ```

---

## Usage

### 1. Build the Pipeline (Ingestion + Indexing)

```bash
python3 scripts/build_pipeline.py
```

**Output:**
- Extracts 167 chunks from `data/` folder
- Builds Chroma index under `vector_db/`
- Creates NetworkX KG snapshot in `kg_store.json`
- Logs progress to stdout

### 2. Start the FastAPI Server

```bash
uvicorn rag_api:app --reload --port 8000
```

**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

Visit http://127.0.0.1:8000/docs for interactive API docs.

### 3. Query the API

#### Simple Vector Lookup
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the possible reasons for rejection of rent receipts?",
    "route": "vector"
  }'
```

#### Multi-Hop Reasoning
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which documents mention both rent receipts and housing loans, and what is the connection?",
    "route": "multihop"
  }'
```

#### Auto-Route (System Decides)
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the process to submit rent receipts for HRA exemption.",
    "route": null
  }'
```

---

## Query Routes

| Route | Use Case | Example |
|-------|----------|---------|
| `vector` | Simple document lookup | "Which PDF mentions Form 13?" |
| `kg` | Entity/relationship queries | "Which documents mention Ascent Consulting?" |
| `hybrid` | Why/how questions | "Why would a rental agreement be rejected?" |
| `multihop` | Complex reasoning across docs | "Which docs mention both X and Y, and how are they related?" |
| `null` | Auto-detect best route | System chooses based on query keywords |

---

## Response Format

```json
{
  "answer": "...",
  "route": "multihop",
  "reasoning_chain": [
    "Step 1: Searching for 'rent receipts'",
    "Step 2: Found 5 initial chunks",
    "Step 3: Extracted related topics: housing loans, HRA exemption",
    "Step 4: Found 3 additional related chunks"
  ],
  "provenance": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "source_file": "User Guide for Investment Proof.pdf",
      "page_number": 12,
      "retrieval_score": 4.92
    }
  ],
  "retrieved_chunks": [ ... ],
  "kg_facts": [ ... ]
}
```

---

## Configuration

Edit `rag_pipeline/config.py` to customize:

```python
@dataclass
class Settings:
    data_dir: Path = Path("data")
    vector_db_dir: Path = Path("vector_db")
    kg_json_path: Path = Path("kg_store.json")
    
    embedding_model: str = "nomic-embed-text:latest"
    llm_model: str = "qwen:latest"
    
    chunk_size: int = 1000
    chunk_overlap: int = 100
    
    top_k: int = 10        # Retrieval candidates
    final_k: int = 5       # Final context chunks
    
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
```

---

## Advanced Features

### Multi-Hop Retrieval

The orchestrator detects queries with keywords like:
- "relationship", "connected", "linked", "how does", "trace", "between", "across"

When detected, it performs:
1. **Initial retrieval** on the query
2. **Topic extraction** using the LLM (from best chunk)
3. **Follow-up retrieval** on extracted topics
4. **Combination** of both results

Example:
```json
{
  "query": "Which documents mention both rent receipts and housing loans, and what is the connection?",
  "route": "multihop"
}
```

### Knowledge Graph Queries

Simple entity extraction and relationship storage:
```python
# Entities extracted from text:
# "Ascent", "Consulting", "Let", "Employee", etc.

# Relationships stored as edges in NetworkX:
# entity1 --[mentioned_in]--> chunk_id
# entity2 --[mentioned_in]--> chunk_id
```

Query with `route: "kg"` to leverage this.

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'rag_pipeline'`

**Solution:** Ensure the repo root is on Python path:
```bash
cd /path/to/RAG
python3 scripts/build_pipeline.py
```

Or set PYTHONPATH:
```bash
PYTHONPATH=. python3 scripts/build_pipeline.py
```

### Error: `ConnectionError: Failed to connect to Ollama`

**Solution:** Start Ollama (macOS):
```bash
ollama serve
```

Or ensure it's running in background. Check:
```bash
curl http://localhost:11434/api/tags
```

### Error: `the input length exceeds the context length`

**Solution:** Already handled via truncation in `rag_pipeline/embeddings.py` (truncates texts to 3000 chars before embedding).

### Chroma `DuplicateIDError`

**Solution:** Already fixed in `rag_pipeline/retriever.py` (deduplicates chunk IDs before adding).

---

## Performance Notes

- **First run (build_pipeline.py)**: ~2-5 minutes (depends on data size + Ollama speed)
  - Document extraction: ~30s
  - Embedding: ~1-2 min (embeddings are slower)
  - KG building: ~30s
  
- **Query latency** (after server is warm):
  - Vector retrieval: 100-500ms
  - LLM response: 2-10s (depends on model + prompt length)

---

## Limits & Future Work

### Current Limitations
- Entity extraction is simple (capitalized words only); use NER/LLM for better results
- No reranking layer (could improve quality with BAAI/bge-reranker-v2-m3)
- No incremental indexing (rebuilds from scratch each time)
- KG uses NetworkX (local memory); use Neo4j for larger graphs

### Planned Enhancements
- ✅ Multi-hop reasoning (implemented)
- ✅ Advanced routing (implemented)
- ✅ Reasoning chain in response (implemented)
- 🔄 Reranker integration (cross-encoder for candidate re-ranking)
- 🔄 Neo4j backend (persistent, scalable KG)
- 🔄 Incremental indexing (file hash tracking)
- 🔄 LangSmith observability (tracing + evaluation)
- 🔄 Fact verification (cross-check answers against sources)

---

## Example Queries

### Basic Lookup
```json
{
  "query": "What is the User Guide for Employee profile?",
  "route": "vector"
}
```

### Reasoning / Why-Question
```json
{
  "query": "Why would a rental agreement be rejected for HRA exemption?",
  "route": "hybrid"
}
```

### Multi-Hop
```json
{
  "query": "Trace the relationship between investment proof submission and tax regime selection across multiple guides.",
  "route": "multihop"
}
```

### Entity Lookup
```json
{
  "query": "Which documents mention Ascent Consulting?",
  "route": "kg"
}
```

### Auto-Route
```json
{
  "query": "Explain the process to submit rent receipts for HRA exemption and what happens if the landlord is your spouse.",
  "route": null
}
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.13 |
| Embeddings | Ollama (nomic-embed-text) |
| LLM | Ollama (qwen) |
| Vector DB | Chroma 1.5.9 |
| BM25 | rank-bm25 |
| KG | NetworkX |
| Framework | LangChain 1.3.1 + LangGraph |
| API | FastAPI |
| PDF Parsing | pypdf |
| Spreadsheet | pandas (openpyxl) |

---

## License

[MIT](LICENSE)

---

## Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes with clear messages
4. Push and open a PR

---

## Contact & Support

For questions or issues, open a GitHub issue in the repository.

---

## Acknowledgments

Built with ❤️ using LangChain, Ollama, Chroma, and NetworkX.

# Employee Knowledge Assistant (EKA)
### Local RAG-based Conversational Assistant for HR, Payroll, Tax, and Employee Documents

---

## Overview

Employee Knowledge Assistant (EKA) is a local Retrieval-Augmented Generation (RAG) application built to help users interact with enterprise documents conversationally.

The system loads PDF documents from a local `data/` directory, retrieves relevant information using embeddings and vector search, and generates grounded responses using a local Large Language Model (LLM).

The solution runs entirely locally using Ollama.

---

# Problem Statement

Organizations often maintain operational knowledge across multiple document formats:

- Employee onboarding guides
- Payroll documentation
- Tax declarations
- HR policies
- Investment proof instructions
- Compliance forms
- Helpdesk documentation
- Employee self-service manuals

These documents become difficult to navigate manually.

Users frequently ask:

- How do I reset my employee portal password?
- How do I submit investment proof?
- What documents are required for leave encashment?
- How do I raise payroll-related queries?
- Which tax forms should be submitted?

Searching manually across documents reduces productivity.

---

# Dataset

## Dataset Source

Local folder:

```text
data/
```

Example document categories:

```text
data/

├── User Guide for Employee profile.pdf
├── Leave_Encashment_Declaration.pdf
├── FAQs on New Tax Regime.pdf
├── Form 10C - PF.pdf
├── Form 13 - PF.pdf
├── Form 12C - IT.pdf
├── Investment Proof Submission Guide.pdf
├── Rent Receipt.pdf
├── Validation Guide.pdf
├── Tax Declaration Forms.pdf
└── Employee Payroll Documents
```

---

# Dataset Characteristics

| Property | Description |
|---|---|
| Format | PDF |
| Content Type | Semi-structured |
| Domain | HR / Payroll / Tax |
| Content | Text + Screenshots + Forms |
| Size | Small → Medium |
| Language | English |
| Layout | Mixed |

---

# Challenges

## 1. Heterogeneous Documents

Documents include:

- Instructions
- Legal declarations
- Tax forms
- UI screenshots
- User manuals

---

## 2. Mixed Information Density

Examples:

- One-page declaration
- 20-page guide
- Structured form

---

## 3. Retrieval Complexity

Keyword search fails because:

```text
Reset Password
↓

Forgot Password
↓

OTP Registration
```

Semantic retrieval is required.

---

## 4. Hallucination Risk

LLMs may invent information if retrieval is weak.

---

## 5. Chunk Boundary Problems

Important context may span pages.

---

# Proposed Solution

Build a Local RAG Pipeline.

Core principles:

- Retrieve before generation
- Source-grounded responses
- Local execution
- No cloud dependency

---

# System Architecture

```text
User Question
      │
      ▼
PDF Loader
      │
      ▼
Document Processing
      │
      ▼
Chunking
      │
      ▼
Embedding
      │
      ▼
Chroma Vector DB
      │
      ▼
Retriever
      │
      ▼
Qwen (LLM)
      │
      ▼
Answer + Sources
```

---

# Technical Pipeline

## Step 1 — Load PDFs

```text
data/*.pdf
```

↓

## Step 2 — Parse Documents

Extract:

- Text
- Metadata
- Page references

↓

## Step 3 — Chunk Documents

Split into semantically searchable sections.

↓

## Step 4 — Generate Embeddings

Convert chunks → vectors.

↓

## Step 5 — Store in Vector DB

ChromaDB.

↓

## Step 6 — Retrieve Context

Top-K relevant chunks.

↓

## Step 7 — Generate Answer

LLM receives:

```text
Question
+
Retrieved Context
```

↓

## Step 8 — Return Response

Output:

```text
Answer
Sources
Confidence
```

---

# Model Selection

## Embedding Model

| Property | Value |
|---|---|
| Name | nomic-embed-text |
| Tag | latest |
| ID | 0a109f422b47 |
| Size | 274 MB |
| Runtime | Ollama |

Purpose:

- Semantic search
- Retrieval
- Context matching

---

## Generation Model

| Property | Value |
|---|---|
| Name | qwen |
| Tag | latest |
| ID | d53d04290064 |
| Size | 2.3 GB |
| Runtime | Ollama |

Purpose:

- Question answering
- Context reasoning
- Response generation

---

# Local Models

Verify:

```bash
ollama list
```

Expected:

```text
NAME                       ID              SIZE

nomic-embed-text:latest    0a109f422b47    274 MB

qwen:latest                d53d04290064    2.3 GB
```

---

# Tech Stack

```text
LangChain
Ollama
Qwen
ChromaDB
PyPDF
Jupyter Notebook
```

---

# Project Structure

```text
employee_knowledge_assistant/

employee_knowledge_assistant_rag.ipynb

requirements.txt

README.md

data/
```

---

# Example Queries

```text
How do I reset password?

How do I declare investments?

What is leave encashment?

Summarize Form 10C.

How can I submit tax proof?
```

---

# Future Enhancements

- Multi-agent retrieval
- Memory
- Streamlit UI
- Evaluation framework
- Observability
- API deployment
- OCR support
- Hybrid Search
- Citation Engine

---

# Success Criteria

✔ Works locally

✔ Answers from PDFs

✔ Source grounded

✔ Single notebook workflow

✔ No external APIs
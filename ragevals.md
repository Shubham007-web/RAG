# RAG & LLM Evaluation Metrics

## Overview

Evaluation metrics for RAG (Retrieval-Augmented Generation) systems can be organized into three major layers:

```text
AI System Evaluation
│
├── Retrieval Metrics
├── Generation Metrics
└── End-to-End RAG Metrics
```

---

# 1. Retrieval Metrics

These metrics evaluate whether the retriever (Vector DB, BM25, Hybrid Search, etc.) returns the right documents.

## 1.1 Precision

### Definition

Out of all retrieved documents, how many are actually relevant?

### Formula

```
Precision = TP / (TP + FP)
```

Where:

- TP = True Positives (Relevant Retrieved)
- FP = False Positives (Irrelevant Retrieved)

### Example

Retrieved Documents = 5

- Relevant = 4
- Irrelevant = 1

```
Precision = 4 / 5 = 0.8
```

### When to Use

- Evaluating Vector Databases
- Comparing Embedding Models
- Reducing retrieval noise

---

## 1.2 Recall

### Definition

Out of all relevant documents, how many did the retriever successfully find?

### Formula

```
Recall = TP / (TP + FN)
```

Where:

- TP = Relevant Retrieved
- FN = Relevant but Missed

### Example

Total Relevant Documents = 10

Retrieved Relevant = 7

```
Recall = 7 / 10 = 0.7
```

### When to Use

- Legal Search
- Medical AI
- Research Assistants

---

## 1.3 Hit Rate (Hit@K)

### Definition

Checks whether at least one relevant document appears within the Top-K retrieved results.

### Formula

```
Hit@K =
(Number of queries with at least one relevant result in Top K)
/
(Total Queries)
```

### Example

100 Queries

92 have at least one correct result in Top-5.

```
Hit@5 = 92 / 100 = 0.92
```

### When to Use

- Initial retriever benchmarking
- Quick comparison between retrieval methods

---

## 1.4 Mean Reciprocal Rank (MRR)

### Definition

Measures how early the first correct document appears.

### Formula

For a single query:

```
RR = 1 / Rank
```

Overall:

```
MRR = (1/N) * Σ(1/Rank_i)
```

### Example

| Query | First Correct Rank |
|--------|-------------------|
| Q1 | 1 |
| Q2 | 2 |
| Q3 | 4 |

```
MRR = (1 + 1/2 + 1/4)/3
     = 0.583
```

### When to Use

- Search Engines
- Recommendation Systems
- Enterprise Search

---

## 1.5 NDCG (Normalized Discounted Cumulative Gain)

### Definition

Rewards highly relevant documents appearing near the top.

Higher-ranked relevant documents contribute more than lower-ranked ones.

### Simplified Idea

```
Higher Relevant Rank
          ↓
Higher NDCG Score
```

### When to Use

- Enterprise Search
- Multi-document Ranking
- Production Retrieval Systems

---

## 1.6 Context Precision (RAGAS)

### Definition

Measures how many retrieved chunks were actually useful.

### Formula

```
Context Precision =
Useful Retrieved Chunks
-----------------------
Total Retrieved Chunks
```

### When to Use

- RAG pipeline evaluation

---

# 2. Generation Metrics

These metrics evaluate the quality of the generated answer.

---

## 2.1 Faithfulness (RAGAS)

### Definition

Checks whether the generated answer is supported by the retrieved context.

### Approximate Formula

```
Faithfulness ≈

Supported Statements
--------------------
Total Statements
```

### Range

- 0 → Hallucinated
- 1 → Fully Grounded

### When to Use

- Every RAG application
- Hallucination detection

---

## 2.2 Answer Relevancy

### Definition

Measures whether the answer actually addresses the user's question.

### Underlying Method

Usually computed using embedding similarity.

```
Question Embedding
        ↕
Answer Embedding
```

### When to Use

- Chatbots
- QA Systems
- Agentic AI

---

## 2.3 Semantic Similarity

### Definition

Measures similarity of meaning rather than exact wording.

### Formula

```
Cosine Similarity =
(A · B)
---------
(||A|| ||B||)
```

Where A and B are embedding vectors.

### Range

- -1 to 1
- Typically:
    - >0.9 Excellent
    - 0.8–0.9 Good
    - <0.7 Weak

### When to Use

- Modern NLP evaluation
- RAG benchmarking

---

## 2.4 BLEU

(Bilingual Evaluation Understudy)

### Definition

Measures n-gram overlap between generated and reference text.

### Simplified Formula

```
Common Words
------------
Generated Words
```

### When to Use

- Machine Translation
- Legacy NLP Systems

### Weakness

Does not capture semantic meaning.

---

## 2.5 ROUGE

(Recall-Oriented Understudy for Gisting Evaluation)

### Definition

Measures overlap with the reference answer.

### Simplified Formula

```
Shared Words
------------
Reference Words
```

### When to Use

- Text Summarization
- QA Benchmarks

---

## 2.6 BERTScore

### Definition

Uses BERT embeddings instead of exact word matching.

### Underlying Method

Computes cosine similarity between contextual token embeddings.

### When to Use

- Modern NLP
- LLM Evaluation
- Semantic Matching

---

# 3. End-to-End RAG Metrics (RAGAS)

```
RAGAS
│
├── Context Precision
├── Context Recall
├── Faithfulness
└── Answer Relevancy
```

---

## 3.1 Context Recall

### Definition

Measures whether retrieval captured all necessary information.

### Formula

```
Context Recall =

Retrieved Relevant Facts
-------------------------
Total Relevant Facts
```

### When to Use

- Detecting missing retrieval information

---

## 3.2 Answer Correctness

### Definition

Measures overall factual correctness.

Some implementations approximate:

```
Answer Correctness =

0.75 × Factuality
+
0.25 × Semantic Similarity
```

*(Implementation may vary across frameworks.)*

---

# 4. LLM-as-a-Judge Metrics

Used by frameworks like DeepEval and G-Eval.

## Common Criteria

- Correctness
- Completeness
- Clarity
- Coherence
- Helpfulness
- Safety
- Conciseness

### Typical Scale

| Score | Meaning |
|---------|----------|
| 1 | Very Poor |
| 2 | Poor |
| 3 | Average |
| 4 | Good |
| 5 | Excellent |

### When to Use

- Production monitoring
- Prompt optimization
- Agent evaluation

---

# 5. Pairwise Evaluation

Instead of assigning absolute scores, compare two responses.

```
Question
│
├── Answer A
└── Answer B
      │
      ▼
  LLM Judge
      │
      ▼
Winner
```

Possible outcomes:

- A > B
- B > A
- Tie

### When to Use

- A/B Testing
- Prompt Engineering
- Model Comparison
- Retriever Comparison

---

# Complete Hierarchy

```text
AI Evaluation
│
├── Retrieval Metrics
│   ├── Precision
│   ├── Recall
│   ├── Hit@K
│   ├── MRR
│   ├── NDCG
│   └── Context Precision
│
├── Generation Metrics
│   ├── Faithfulness
│   ├── Answer Relevancy
│   ├── Semantic Similarity
│   ├── BLEU
│   ├── ROUGE
│   └── BERTScore
│
├── End-to-End RAG Metrics
│   ├── Context Precision
│   ├── Context Recall
│   ├── Faithfulness
│   └── Answer Relevancy
│
└── LLM-as-a-Judge
    ├── Correctness
    ├── Completeness
    ├── Clarity
    ├── Coherence
    ├── Helpfulness
    ├── Safety
    └── Pairwise Evaluation
```

---

# Industry Cheat Sheet

| Scenario | Recommended Metrics |
|------------|------------------|
| Vector DB Evaluation | Precision, Recall, Hit@K, MRR |
| Enterprise Search | MRR, NDCG |
| RAG Chatbot | Faithfulness, Context Precision, Context Recall |
| Hallucination Detection | Faithfulness |
| Prompt Comparison | Pairwise Evaluation |
| Model Benchmarking | Pairwise + Correctness |
| Translation | BLEU |
| Summarization | ROUGE |
| Modern LLM Evaluation | Semantic Similarity, BERTScore |
| Production Agentic AI | RAGAS + LLM-as-a-Judge + Pairwise |
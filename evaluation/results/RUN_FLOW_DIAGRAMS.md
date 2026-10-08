# 📊 Case Study: End-to-End Flow Diagrams for Every Experimental Run

**Project:** Autonomous RAG Pipeline Benchmark & Case Study  
**Corpus:** 13 Multimodal Vision-Language & Embodied AI Research Papers  
**Benchmark:** 122 Curated Golden Questions across 8 Taxonomy Classes  

This document provides dedicated, production-grade architecture flow diagrams for **every experimental run** analyzed in the RAG Case Study ([RAG_CASE_STUDY.md](file:///home/shyam/rag/evaluation/results/RAG_CASE_STUDY.md)) and tracked in the Experiment Registry ([EXPERIMENTS.md](file:///home/shyam/rag/evaluation/results/EXPERIMENTS.md)).

---

## 🗺️ Master Architecture Overview

The diagram below maps the complete RAG lifecycle and shows where each experimental run intervenes:

```mermaid
flowchart TB
    subgraph S1["Stage 1: Document Ingestion"]
        PDF["13 Raw PDFs (data/raw)"] --> Loader["PDF Loader (PyMuPDF / pdf_loader.py)"]
        Loader --> Cleaner["Text Cleaner (cleaner.py)"]
    end

    subgraph S2["Stage 2: Chunking Strategy (Run 0 vs Run 1)"]
        Cleaner --> Chunker{"Chunking Selection"}
        Chunker -->|"Run 0 (Baseline)"| ChkSem["Semantic Chunking"]
        Chunker -->|"Run 1A"| ChkRec["Recursive Character Chunking"]
        Chunker -->|"Run 1B"| ChkFix["Fixed-Window Chunking"]
        Chunker -->|"Run 1C"| ChkMark["Markdown-Aware Chunking"]
    end

    subgraph S3["Stage 3: Embedding & Indexing (Run 0 vs Run 5)"]
        ChkSem --> EmbedChoice{"Embedding Model"}
        ChkRec --> EmbedChoice
        ChkFix --> EmbedChoice
        ChkMark --> EmbedChoice
        EmbedChoice -->|"Run 0 / 1 / 2 / 3 / 4"| EmbQwen["Qwen3-Embedding-0.6B (1024-d)"]
        EmbedChoice -->|"Run 5 (Scale)"| EmbBGE["BGE-M3 / GTE-Large"]
        EmbQwen --> Qdrant[("Qdrant Vector DB")]
        EmbBGE --> Qdrant
    end

    subgraph S4["Stage 4: Retrieval & Fusion (Run 0 vs Run 2 vs Run 3)"]
        Query["User / Benchmark Query"] --> RetChoice{"Retrieval Mode"}
        RetChoice -->|"Run 0 (Baseline)"| RetTop5["Dense Search: Top-k = 5"]
        RetChoice -->|"Run 2 (Depth)"| RetTopK["Dense Search: Top-k = 10 / 15"]
        RetChoice -->|"Run 3 (Hybrid)"| RetHybrid["Dense (Qdrant) + Sparse (BM25)"]
        Qdrant -.-> RetTop5
        Qdrant -.-> RetTopK
        Qdrant -.-> RetHybrid
    end

    subgraph S5["Stage 5: Re-ranking (Run 0 vs Run 4)"]
        RetTop5 --> PassThru["Direct Pass-Through"]
        RetTopK --> PassThru
        RetHybrid --> PassThru
        RetTopK -->|"Run 4"| CrossEnc["BGE-Reranker-Large (Cross-Encoder)"]
        CrossEnc --> RerankedTop5["Re-ranked Top-5 Chunks"]
    end

    subgraph S6["Stage 6: Prompt & Generation"]
        PassThru --> PromptEngine["Prompt Formatter (Context + Query)"]
        RerankedTop5 --> PromptEngine
        PromptEngine --> LLM["Qwen3.5:4B LLM (num_ctx: 8192, reasoning: false)"]
        LLM --> GenAnswer["Generated Grounded Answer"]
    end

    subgraph S7["Stage 7: Autonomous Diagnostic Evaluation"]
        GenAnswer --> EvalSuite["Automated Evaluator (evaluate.py)"]
        EvalSuite --> FailureTaxonomy["Diagnostic Taxonomy (Cat A, B, C, D)"]
    end
```

---

## 🔒 Run 0: Baseline Architecture (Frozen Benchmark)

**Configuration:** Semantic Chunking + `Qwen3-Embedding-0.6B` + Qdrant Top-5 + `Qwen3.5:4B` (`num_ctx: 8192`, `reasoning: false`, `num_predict: 1024`).  
**Core Metric Anchor:** **98.4%** Top-5 Document Hit | **82.0%** LLM Completion Rate | **0%** Cutoff Rate.

```mermaid
flowchart TD
    subgraph INGEST["1. Ingestion & Preprocessing"]
        PDF["13 Research Papers (data/raw/*.pdf)"] --> EXT["extract_pdf() (Page-by-Page)"]
        EXT --> CLN["clean_text() (Whitespace & Hyphen Normalization)"]
    end

    subgraph CHUNK["2. Semantic Chunking"]
        CLN --> SEM["create_chunks(strategy='semantic')"]
        SEM --> PAYLOAD["Payload Construction (chunk_id, paper_id, page, text)"]
    end

    subgraph EMBED_INDEX["3. Embedding & Vector Storage"]
        PAYLOAD --> EMB["Embedder.embed_chunks() using Qwen3-Embedding-0.6B"]
        EMB --> QDRANT[("Qdrant Collection: Qwen3_Embedding_0_6B_semantic (Cosine Distance)")]
    end

    subgraph INFERENCE["4. Online Retrieval & Generation Pipeline"]
        GOLDEN_Q["Benchmark Question (122 Golden Qs)"] --> QUERY_EMB["Embedder.embed_query(query)"]
        QUERY_EMB --> SEARCH["QdrantClient.query_points(limit=5)"]
        QDRANT -.-> SEARCH
        SEARCH --> CHUNKS["Top-5 Retrieved Context Chunks"]
        CHUNKS --> FORMAT["Format Prompt:\n'Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer using only context.'"]
        FORMAT --> OLLAMA["ChatOllama(model='qwen3.5:4b', reasoning=False, num_ctx=8192)"]
        OLLAMA --> RESP["Final Generated Answer"]
    end

    subgraph BENCHMARK["5. Quantitative Benchmark Logging"]
        RESP --> COMP["evaluate.py Benchmark Engine"]
        COMP --> METRICS["Metrics Computed:\n- Top-5 Doc Hit: 98.4%\n- Top-1 Doc Hit: 90.2%\n- Page Hit ±1: 71.3%\n- Exact Page Hit: 52.5%\n- Completed Ans: 82.0%\n- Refusal Rate: 18.0%"]
    end
```

---

## 📋 Run 1: Chunking Strategies Comparison Flow

**Objective:** Compare 4 chunking strategies to eliminate page fragmentation and solve long-document retrieval issues (such as Flamingo's low 33.3% page hit in baseline).

```mermaid
flowchart TD
    RAW["Raw Document Stream (data/raw/*.pdf)"] --> CLEAN["Text Cleaning & Normalization"]

    CLEAN --> S1["Strategy 1A: Fixed-Size Chunking"]
    CLEAN --> S2["Strategy 1B: Recursive Character Chunking"]
    CLEAN --> S3["Strategy 1C: Markdown-Aware Structure Chunking"]
    CLEAN --> S4["Strategy 1D: Semantic Chunking (Run 0 Anchor)"]

    subgraph SPEC_1A["Fixed-Size (500 tokens / 50 overlap)"]
        S1 --> P1A["Strict character/token window slicing.\nNo semantic or paragraph awareness."]
        P1A --> C1A["Pros: Uniform chunk distribution.\nCons: Frequently splits tables and equations mid-sentence."]
    end

    subgraph SPEC_1B["Recursive Splitting (500 tokens / 50 overlap)"]
        S2 --> P1B["Hierarchical separators:\n['\n\n', '\n', ' ', '']"]
        P1B --> C1B["Pros: Preserves natural paragraphs.\nCons: Arbitrary breaks across section transitions."]
    end

    subgraph SPEC_1C["Markdown / Header-Aware Chunking"]
        S3 --> P1C["Splits on document structural markers:\n[# H1, ## H2, ### H3, Table Blocks]"]
        P1C --> C1C["Pros: Captures complete tables & methodology sections.\nCons: Variable chunk lengths (skewed vector densities)."]
    end

    subgraph SPEC_1D["Semantic Chunking"]
        S4 --> P1D["Calculates cosine distance between sentence embeddings.\nSplits when semantic distance exceeds threshold."]
        P1D --> C1D["Pros: High conceptual cohesion per chunk.\nCons: Slower ingestion, variable chunk size."]
    end

    C1A --> EVAL_COMP["Comparative Evaluation Matrix on 122 Golden Questions"]
    C1B --> EVAL_COMP
    C1C --> EVAL_COMP
    C1D --> EVAL_COMP

    EVAL_COMP --> OUT["Metric Delta Analysis:\n- Exact Page Hit Rate\n- Table Parsing Accuracy\n- Context Boundary Fragmentation"]
```

---

## 📋 Run 2: Retrieval Depth (Top-$k$ Scaling: 5 vs 10 vs 15)

**Objective:** Test if increasing retrieval depth from $k=5$ to $k=10$ or $k=15$ recovers the 21 missing evidence chunks (Category A failures), without overwhelming the LLM with irrelevant survey distractors.

```mermaid
flowchart TD
    Q["Benchmark Query"] --> EMB["Embed Query (Qwen3-0.6B)"]
    EMB --> VEC_SEARCH["Qdrant Vector Search"]

    subgraph PATH_A["Baseline Depth (k = 5)"]
        VEC_SEARCH --> K5["Retrieve Top-5 Chunks (~2,500 tokens)"]
        K5 --> CTX5["Pack into Prompt (Context Usage: ~30%)"]
        CTX5 --> LLM5["Qwen3.5:4B Inference"]
        LLM5 --> OUT5["Result:\n- Top-5 Doc Hit: 98.4%\n- Refusal Rate: 18.0% (21 Cat A misses)"]
    end

    subgraph PATH_B["Expanded Depth (k = 10)"]
        VEC_SEARCH --> K10["Retrieve Top-10 Chunks (~5,000 tokens)"]
        K10 --> CTX10["Pack into Prompt (Context Usage: ~60%)"]
        CTX10 --> LLM10["Qwen3.5:4B Inference"]
        LLM10 --> OUT10["Tradeoff:\n+ Recovers missing evidence in long papers\n- Increases prompt latency\n- Risk of 'Lost in the Middle'"]
    end

    subgraph PATH_C["Deep Depth (k = 15)"]
        VEC_SEARCH --> K15["Retrieve Top-15 Chunks (~7,500 tokens)"]
        K15 --> CTX15["Pack into Prompt (Context Usage: ~90% of 8192)"]
        CTX15 --> LLM15["Qwen3.5:4B Inference"]
        LLM15 --> OUT15["Tradeoff:\n+ Maximum possible retrieval recall\n- High distraction from Survey/LLaVA attractor chunks"]
    end
```

---

## 📋 Run 3: Hybrid Search (Dense Vector + BM25 Sparse Keyword Fusion)

**Objective:** Combine dense semantic retrieval with sparse lexical BM25 retrieval to fix keyword/numerical misses (e.g., exact model parameters, dataset numbers where baseline completed answers was only 54.5%).

```mermaid
flowchart TD
    QUERY["Query: e.g. 'What is the image resolution used for RefCOCOg evaluation in KOSMOS-2?'"]

    subgraph DENSE_BRANCH["Dense Semantic Search Branch"]
        QUERY --> D_EMB["Qwen3-Embedding-0.6B"]
        D_EMB --> D_VEC["Dense Query Vector (1024-d)"]
        D_VEC --> D_QDRANT[("Qdrant Dense Index")]
        D_QDRANT --> D_TOP["Top-15 Dense Candidates with Cosine Scores"]
    end

    subgraph SPARSE_BRANCH["Sparse Lexical Keyword Branch"]
        QUERY --> S_TOK["BM25 Tokenizer & Stemmer ('resolution', '224x224', 'RefCOCOg')"]
        S_TOK --> S_INDEX[("BM25 Inverted Term Index")]
        S_INDEX --> S_TOP["Top-15 Sparse Candidates with BM25 Scores"]
    end

    subgraph FUSION["Reciprocal Rank Fusion (RRF)"]
        D_TOP --> RRF["RRF Algorithm:\nRRF_Score(d) = sum( 1 / (60 + rank_dense(d)) + 1 / (60 + rank_bm25(d)) )"]
        S_TOP --> RRF
        RRF --> COMBINED_RANK["Fused & De-duplicated Top-5 Chunks"]
    end

    subgraph GENERATION["Context-Grounded Generation"]
        COMBINED_RANK --> PROMPT["Context Assembly"]
        PROMPT --> QWEN["Qwen3.5:4B Generator"]
        QWEN --> ANS["High-Precision Answer (Exact Numbers Restored)"]
    end
```

---

## 📋 Run 4: Two-Stage Re-ranking Flow (Bi-Encoder + Cross-Encoder)

**Objective:** Mitigate literature survey attractors (e.g. Survey Paper or LLaVA displacing primary paper chunks) by scoring `[Query, Chunk]` pairs jointly with a full cross-attention model (`BGE-Reranker-Large`).

```mermaid
flowchart TD
    QUERY["User / Benchmark Query"] --> EMB["Embed Query (Bi-Encoder)"]

    subgraph STAGE1["Stage 1: High-Recall First Stage Retrieval (Bi-Encoder)"]
        EMB --> QDRANT[("Qdrant Vector Database")]
        QDRANT --> CANDIDATES["Retrieve Top-15 Candidate Chunks\n(Fast Vector Distance, Cosine ~0.70-0.78)"]
        CANDIDATES --> NOTICE["Issue in Baseline: Literature review papers (e.g. Survey) frequently outscore primary paper chunks"]
    end

    subgraph STAGE2["Stage 2: High-Precision Cross-Encoder Re-ranking"]
        QUERY -.-> PAIRS["Construct 15 (Query, Chunk) Text Pairs"]
        NOTICE --> PAIRS
        PAIRS --> RERANKER["BGE-Reranker-Large (Cross-Encoder)\nFull Cross-Attention over (Query, Chunk)"]
        RERANKER --> RE_SCORED["Compute Logit Relevance Scores"]
        RE_SCORED --> SORT["Re-rank and Select Top-5 Chunks"]
    end

    subgraph OUTCOME["Stage 3: Targeted Context Grounding"]
        SORT --> FILTERED["Top-5 Filtered Chunks\n(Survey distractors demoted; primary target promoted)"]
        FILTERED --> PROMPT["Format Context into System Prompt"]
        PROMPT --> LLM["Qwen3.5:4B LLM"]
        LLM --> FINAL["Accurate, Source-Specific Answer"]
    end
```

---

## 📋 Run 5: Embedding Model Scaling & Representation Flow

**Objective:** Compare embedding representation models to evaluate how vector dimensionality, multi-lingual capabilities, and architecture size affect retrieval hit rates.

```mermaid
flowchart TD
    TEXT["Ingested Chunk Text (500 tokens)"]

    subgraph EMBEDDERS["Embedding Model Candidates"]
        TEXT --> M1["Candidate A: Qwen3-Embedding-0.6B\n(Current Baseline: 1024-dim, Fast, Local)"]
        TEXT --> M2["Candidate B: BGE-M3\n(1024-dim, Multi-Lingual, Dense + Sparse + ColBERT Multi-Vector)"]
        TEXT --> M3["Candidate C: GTE-Large\n(1024-dim, High-Capacity Dual Encoder)"]
    end

    subgraph VECTOR_SPACES["Qdrant Separate Index Collections"]
        M1 --> COLL1[("Collection: Qwen3_Embedding_0_6B_semantic")]
        M2 --> COLL2[("Collection: BGE_M3_semantic")]
        M3 --> COLL3[("Collection: GTE_Large_semantic")]
    end

    subgraph BENCHMARK_EVAL["Parallel Retrieval Evaluation across 122 Golden Questions"]
        COLL1 --> RES1["Top-5 Doc Hit: 98.4%\nAvg Top-1 Sim: 0.7339"]
        COLL2 --> RES2["Target: Test Multi-Vector Disambiguation"]
        COLL3 --> RES3["Target: Test Fine-Grained Technical Clustering"]
    end
```

---

## 🧪 Automated Failure Classification & Diagnostic Loop

The evaluation engine automatically audits every question and categorizes errors into four actionable buckets:

```mermaid
flowchart TD
    START["Benchmark Question Run (evaluate.py)"] --> RET["Retrieve Top-5 Chunks from Qdrant"]
    RET --> CHK_DOC{"Did any chunk match target document?"}

    CHK_DOC -->|"No"| CATA1["Category A: Retrieval Failure (Target Doc Missed)"]
    CHK_DOC -->|"Yes"| CHK_PG{"Did any chunk contain the target page (±1)?"}

    CHK_PG -->|"No"| CATA2["Category A: Retrieval Failure (Page Missed)"]
    CHK_PG -->|"Yes"| LLM_RUN["Prompt Qwen3.5:4B LLM with Top-5 Chunks"]

    LLM_RUN --> CHK_REFUSAL{"Did LLM output refusal?\n('not mentioned / not in context')"}
    CHK_REFUSAL -->|"Yes"| CHK_EVIDENCE{"Was actual answer present in chunk text?"}

    CHK_EVIDENCE -->|"Yes"| CATB["Category B: Context Present, but LLM Failed to Extract"]
    CHK_EVIDENCE -->|"No"| CATA3["Category A: Faithful Refusal due to Missing Evidence"]

    CHK_REFUSAL -->|"No"| CHK_TRUTH{"Does generated answer match golden expected answer?"}
    CHK_TRUTH -->|"Yes"| PASS["✅ Passing: Grounded & Accurate (53.3% in Baseline)"]
    CHK_TRUTH -->|"No"| CHK_AMBIG{"Is question ambiguous or requires multi-page synthesis?"}

    CHK_AMBIG -->|"Yes"| CATD["Category D: Question Ambiguity / Multi-Hop Requirement"]
    CHK_AMBIG -->|"No"| CATC["Category C: LLM Hallucination / Ungrounded Claim"]

    subgraph REMEDIES["Actionable Engineering Interventions"]
        CATA1 & CATA2 & CATA3 -.-> FIX_A["Fix with Run 2 (Top-10), Run 3 (Hybrid Search), or Run 1 (Chunking)"]
        CATB -.-> FIX_B["Fix with Structured Prompt Engineering & Table Formatting"]
        CATC -.-> FIX_C["Fix with Zero-Speculation Prompt & Run 4 (Cross-Encoder Re-ranking)"]
        CATD -.-> FIX_D["Fix with Multi-Hop Query Decomposition"]
    end
```

---

## 📌 Summary Reference Table of All Runs

| Run # | Experiment Name | Primary Variable Changed | Upstream Dependency | Key Problem Solved |
| :---: | :--- | :--- | :--- | :--- |
| **0** | **Baseline (Frozen)** | Control Anchor | Semantic + Qwen3-0.6B + Qdrant Top-5 + Qwen3.5:4B | Foundation benchmark (98.4% Doc Hit, 82.0% Ans) |
| **1** | **Chunking Strategies** | Chunking Algorithm (Recursive / Fixed / Markdown / Semantic) | Ingestion & `chunking.py` | Eliminates table fragmentation & improves page precision |
| **2** | **Retrieval Depth** | Top-$k$ parameter ($k=5 \rightarrow 10 \rightarrow 15$) | `evaluate.py` retriever limit | Recovers missing evidence chunks for long papers |
| **3** | **Hybrid Search** | Dual-channel retrieval (Dense Cosine + Sparse BM25 + RRF) | Sparse index + Qdrant | Resolves numerical & exact keyword token misses |
| **4** | **Re-ranking** | Cross-Encoder (`BGE-Reranker-Large`) over Top-15 | Two-stage inference pipeline | Discards literature survey attractors; promotes primary papers |
| **5** | **Embedding Scaling** | Vector Model (`Qwen3-0.6B` vs `BGE-M3` vs `GTE-Large`) | Model embedding dimension & weights | Evaluates vector representation geometry & capacity |

---

*File generated for RAG Case Study documentation. Linked from [RAG_CASE_STUDY.md](file:///home/shyam/rag/evaluation/results/RAG_CASE_STUDY.md) and [EXPERIMENTS.md](file:///home/shyam/rag/evaluation/results/EXPERIMENTS.md).*

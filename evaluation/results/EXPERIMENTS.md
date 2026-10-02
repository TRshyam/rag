# 🧪 Autonomous RAG Experiment Tracker & Registry

This registry tracks systematic experiments evaluating retrieval quality, chunking strategies, embedding representations, and generation fidelity across 13 research papers (122 benchmark golden questions).

---

## 📌 Experiment Registry Table

| Exp # | Experiment Name | Status | Chunking | Embedding | Vector DB | LLM | Top-k | Top-5 Hit | Answer % | Key Finding / Focus |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **Baseline** | 🔒 **FROZEN** | Semantic | Qwen3-0.6B | Qdrant | Qwen3.5:4B (`reasoning: false`, `ctx: 8192`) | 5 | **98.4%** | **82.0%** | Baseline anchor: 0 token cutoffs, 100/122 completed answers, 22 faithful refusals |
| *1* | *Chunking Strategies* | 📋 Planned | Recursive / Fixed / Markdown | Qwen3-0.6B | Qdrant | Qwen3.5:4B | 5 | TBD | TBD | Test if recursive or markdown chunking improves page precision on long papers (Flamingo) |
| *2* | *Retrieval Depth (Top-K)* | 📋 Planned | Semantic | Qwen3-0.6B | Qdrant | Qwen3.5:4B | 10 | TBD | TBD | Measure recall boost on dense evidence questions vs noise injection |
| *3* | *Hybrid Search (Dense + BM25)* | 📋 Planned | Semantic | Qwen3-0.6B + BM25 | Qdrant | Qwen3.5:4B | 5 | TBD | TBD | Mitigate survey hub attractors and exact numerical keyword misses |
| *4* | *Re-ranking (Cross-Encoder)* | 📋 Planned | Semantic | Qwen3-0.6B | Qdrant | Qwen3.5:4B | 15 &rarr; 5 | TBD | TBD | Re-rank candidates using BGE-Reranker-Large to elevate target primary paper chunks |
| *5* | *Embedding Model Scale* | 📋 Planned | Semantic | BGE-M3 / GTE | Qdrant | Qwen3.5:4B | 5 | TBD | TBD | Compare 0.6B embedding with larger bi-encoders |

---

## 🔒 Experiment 0: Baseline (Frozen Configuration)

**Freeze Date:** October 2, 2026  
**Artifacts:**
- Configuration: [`evaluation/configs/baseline_experiment_0.json`](file:///home/shyam/rag/evaluation/configs/baseline_experiment_0.json)
- Evaluation Results JSON: [`evaluation/results/results data/experiment_0_baseline.json`](file:///home/shyam/rag/evaluation/results/results%20data/experiment_0_baseline.json)
- Markdown Summary: [`evaluation/results/evaluation_summary.md`](file:///home/shyam/rag/evaluation/results/evaluation_summary.md)
- Case Study Document: [`evaluation/results/RAG_CASE_STUDY.md`](file:///home/shyam/rag/evaluation/results/RAG_CASE_STUDY.md)
- Interactive Dashboard: [`evaluation/results/report.html`](file:///home/shyam/rag/evaluation/results/report.html)

### Configuration Specification
```text
BASELINE
────────────────────────────
Chunking:    Semantic
Embedding:   Qwen3-Embedding-0.6B
Vector DB:   Qdrant
LLM:         Qwen3.5:4B
num_ctx:     8192
reasoning:   false
num_predict: 1024

Questions:   122
Top-5 Hit:   98.4%
Answer:      82.0%
```

### Quantitative Metrics Summary

| Metric Category | Metric Name | Baseline Value | Interpretation |
| :--- | :--- | :---: | :--- |
| **Corpus & Benchmark** | Corpus Size | 13 Papers | All 13 raw research papers ingested & indexed in Qdrant |
| | Evaluated Papers | 10 Papers | 10 papers evaluated with high-fidelity golden Q&A dataset |
| | Benchmark Questions | 122 Qs | Multi-domain questions across 8 taxonomy categories |
| **Retrieval Efficacy** | Top-5 Target Document Hit | **98.4% (120/122)** | Retriever places correct document in top 5 chunks in almost every query |
| | Top-1 Target Document Hit | **90.2% (110/122)** | Top-ranked chunk matches target document in 9 out of 10 queries |
| | Top-5 Target Page Hit (±1 Pg) | **71.3% (87/122)** | Exact target page or adjacent page context retrieved in top 5 |
| | Top-5 Exact Page Hit | **52.5% (64/122)** | Exact target page retrieved in top 5 |
| | Average Top-1 Similarity | **0.7339** | Mean cosine similarity for rank-1 retrieved chunk |
| | Average All-Chunks Similarity | **0.6900** | Mean cosine similarity across all 5 retrieved chunks |
| **Generation Fidelity** | LLM Completed Answers | **82.0% (100/122)** | Clean, grounded answers matching golden truth |
| | LLM Cutoff / Empty Answers | **0.0% (0/122)** | 0 token cutoffs after disabling reasoning scratchpad |
| | LLM Refusals (Not in Context)| **18.0% (22/122)** | Model strictly declined to hallucinate when context was insufficient |
| **Automated Diagnostic Taxonomy** | **Cat A: Retrieval Failure** | **21 Qs (17.2%)** | Evidence missing from Top-5 chunks; LLM faithfully refused |
| | **Cat B: Context Present, LLM Failed** | **20 Qs (16.4%)** | Evidence present in chunks, but LLM missed or misread it |
| | **Cat C: LLM Hallucination** | **13 Qs (10.7%)** | LLM asserted ungrounded/fabricated facts or incorrect metrics |
| | **Cat D: Question Ambiguity** | **3 Qs (2.5%)** | Ambiguous question or requires multi-page synthesis |
| | **Passing / Fully Grounded** | **65 Qs (53.3%)** | High-fidelity extraction matching golden answer |

### Automated Failure Taxonomy (A, B, C, D) Breakdown

| Category | Classification | Count | % of All | % of Errors | Actionable Engineering Remedy |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **A** | **Retrieval failure** | **21** | **17.2%** | **36.8%** | Expand retrieval depth to Top-10; add BM25 keyword matching (Hybrid Search); benchmark alternative chunking strategies. |
| **B** | **Context contains answer, but LLM failed** | **20** | **16.4%** | **35.1%** | Refine prompt template to guide table parsing; add chunk ranking / source highlight markers; increase extraction clarity. |
| **C** | **LLM hallucination** | **13** | **10.7%** | **22.8%** | Strengthen zero-speculation prompt instructions; enforce explicit citation matching before emitting assertions. |
| **D** | **Question ambiguity/difficulty** | **3** | **2.5%** | **5.3%** | Refine benchmark question wording in dataset or implement multi-hop query decomposition. |

### Per-Paper Performance Breakdown

| Research Paper | Benchmark Qs | Top-5 Doc Hit (%) | Top-5 Page Hit (±1) (%) | Top-1 Doc Hit (%) | Avg Top-1 Score | Completed LLM Ans (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BLIP-2** | 13 | 100.0% (13/13) | 76.9% (10/13) | 100.0% | 0.7509 | 84.6% (11/13) |
| **CLIP** | 12 | 100.0% (12/12) | 75.0% (9/12) | 100.0% | 0.7598 | 58.3% (7/12) |
| **Flamingo** | 12 | 91.7% (11/12) | 33.3% (4/12) | 75.0% | 0.7261 | 91.7% (11/12) |
| **GLIP** | 12 | 100.0% (12/12) | 58.3% (7/12) | 91.7% | 0.7185 | 75.0% (9/12) |
| **GroundCount** | 12 | 100.0% (12/12) | 83.3% (10/12) | 100.0% | 0.7326 | 91.7% (11/12) |
| **Grounding DINO** | 12 | 100.0% (12/12) | 66.7% (8/12) | 100.0% | 0.7849 | 83.3% (10/12) |
| **KOSMOS-2** | 12 | 91.7% (11/12) | 66.7% (8/12) | 83.3% | 0.6807 | 75.0% (9/12) |
| **PaLM-E** | 12 | 100.0% (12/12) | 75.0% (9/12) | 83.3% | 0.7555 | 66.7% (8/12) |
| **RT-2** | 12 | 100.0% (12/12) | 75.0% (9/12) | 91.7% | 0.7306 | 91.7% (11/12) |
| **Thinking with Visual Grounding** | 13 | 100.0% (13/13) | 100.0% (13/13) | 92.3% | 0.7004 | 100.0% (13/13) |

### Per-Question Category Performance Breakdown

| Category | Questions | Top-5 Doc Hit (%) | Top-5 Page Hit (±1) (%) | Avg Top-1 Score | Completed LLM Ans (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Architecture** | 12 | 100.0% (12/12) | 75.0% (9/12) | 0.7700 | 100.0% (12/12) |
| **Comparison** | 12 | 100.0% (12/12) | 66.7% (8/12) | 0.7414 | 83.3% (10/12) |
| **Definition** | 18 | 94.4% (17/18) | 55.6% (10/18) | 0.6968 | 77.8% (14/18) |
| **Numbers** | 11 | 100.0% (11/11) | 100.0% (11/11) | 0.7126 | 54.5% (6/11) |
| **Performance** | 13 | 100.0% (13/13) | 69.2% (9/13) | 0.7781 | 69.2% (9/13) |
| **Procedure** | 35 | 97.1% (34/35) | 74.3% (26/35) | 0.7322 | 85.7% (30/35) |
| **Reason/Why** | 17 | 100.0% (17/17) | 70.6% (12/17) | 0.7309 | 94.1% (16/17) |
| **Training** | 4 | 100.0% (4/4) | 50.0% (2/4) | 0.7107 | 75.0% (3/4) |

---

## 🔬 Reproducibility Commands

To re-run or inspect this frozen baseline:

```bash
# 1. Run full baseline evaluation (122 questions)
python evaluation/evaluate.py --all

# 2. Inspect results in CLI terminal
python evaluation/view_results.py --summary

# 3. Launch or generate Interactive Dashboard & Markdown Report
python evaluation/view_results.py --file "evaluation/results/results data/experiment_0_baseline.json"
```

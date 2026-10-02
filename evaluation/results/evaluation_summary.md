# 📊 RAG Evaluation Results Summary — Experiment 0 / Baseline

```
BASELINE (FROZEN CONFIGURATION)
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

**Pipeline**: Qwen3-Embedding-0.6B &rarr; Qdrant (`Qwen3_Embedding_0_6B_semantic`) &rarr; Qwen3.5:4B

**Corpus Size**: 13 Research Papers (10 Evaluated · 3 Context Chunks)

**Total Benchmark Questions**: 122

---

## 1. Executive Performance Metrics

| Metric | Count | Rate (%) | Description |
| :--- | :---: | :---: | :--- |
| **Corpus Coverage** | 10/13 Papers | **76.9%** | 10 papers tested with golden questions; 3 papers indexed in Qdrant |
| **Top-5 Target Document Hit** | 120/122 | **98.4%** | Target research paper retrieved in top 5 chunks |
| **Top-1 Target Document Hit** | 110/122 | **90.2%** | Target research paper is the #1 ranked chunk |
| **Top-5 Target Page Hit (±1 Pg)** | 87/122 | **71.3%** | Retrieved chunk from exact target page or ±1 page |
| **Top-5 Exact Page Hit** | 64/122 | **52.5%** | Retrieved chunk from exact target page |
| **Average Top-1 Similarity Score** | - | **0.7339** | Mean cosine similarity of top retrieved chunk |
| **Average All-Chunks Score** | - | **0.6900** | Mean similarity across all 5 retrieved chunks |
| **LLM Completed Answers** | 100/122 | **82.0%** | Successfully generated answers |
| **LLM Cutoff / Empty Answers** | 0/122 | **0.0%** | Model stopped inside reasoning `<think>` block or hit token limit |
| **LLM Refusals (Not Found)** | 22/122 | **18.0%** | Model stated info is not mentioned in retrieved context |

## 2. Automated Failure Taxonomy Breakdown (A, B, C, D)

| Category | Classification | Count | % of All Qs | Description & Recommended Remedy |
| :---: | :--- | :---: | :---: | :--- |
| **A** | **Retrieval failure** | **21** | **17.2%** | Target chunk/evidence was missing from Top-K; LLM correctly refused. *Remedy: Increase Top-K (5 &rarr; 10), add BM25 hybrid search, test Markdown/Recursive chunking.* |
| **B** | **Context contains answer, but LLM failed** | **20** | **16.4%** | Ground truth evidence was present in retrieved chunks, but LLM missed or misread it. *Remedy: Refine extraction prompt, add chunk re-ranking or citation tagging.* |
| **C** | **LLM hallucination** | **13** | **10.7%** | LLM asserted ungrounded/fabricated facts or incorrect metrics. *Remedy: Enforce strict grounding constraints and zero-speculation directive.* |
| **D** | **Question ambiguity/difficulty** | **3** | **2.5%** | Question suffers from ambiguity or requires distant multi-page synthesis. *Remedy: Benchmark question refinement or multi-hop retrieval.* |
| **PASS** | **Correct / Grounded Answers** | **65** | **53.3%** | Fully grounded and factually aligned with golden answers. |

## 3. Research Papers Corpus Overview (All 13 Papers)

| # | Research Paper | Evaluation Status | Benchmark Questions | Retrieved Chunks | Source File |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **BLIP-2** | ✅ Evaluated | 13 Qs | 55x | `BLIP-2_Bootstrapping_Language-Image_Pre-training_w.pdf` |
| 2 | **CLIP** | ✅ Evaluated | 12 Qs | 57x | `Learning Transferable Visual Models From Natural Language Supervision.pdf` |
| 3 | **Flamingo** | ✅ Evaluated | 12 Qs | 57x | `Flamingo: a Visual Language Model for Few-Shot Learning.pdf` |
| 4 | **GLIP** | ✅ Evaluated | 12 Qs | 60x | `Grounded Language-Image Pre-training.pdf` |
| 5 | **GroundCount** | ✅ Evaluated | 12 Qs | 52x | `Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations.pdf` |
| 6 | **Grounding DINO** | ✅ Evaluated | 12 Qs | 62x | `Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.pdf` |
| 7 | **KOSMOS-2** | ✅ Evaluated | 12 Qs | 37x | `KOSMOS-2: Grounding Multimodal Large Language Models to the World.pdf` |
| 8 | **PaLM-E** | ✅ Evaluated | 12 Qs | 51x | `PaLM-E: An Embodied Multimodal Language Model.pdf` |
| 9 | **RT-2** | ✅ Evaluated | 12 Qs | 71x | `RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Contro.pdf` |
| 10 | **Thinking with Visual Grounding** | ✅ Evaluated | 13 Qs | 75x | `Thinking with Visual Grounding.pdf` |
| 11 | **VIMA** | ⚠️ In Corpus (Context Only) | 0 Qs (Unassessed) | 9x | `VIMA: General Robot Manipulation with Multimodal Prompts.pdf` |
| 12 | **Vision Language Models Survey** | ⚠️ In Corpus (Context Only) | 0 Qs (Unassessed) | 7x | `Vision Language Models: A Survey of 26K Papers (CVPR, ICLR, NeurIPS 2023–2025).pdf` |
| 13 | **Visual Instruction Tuning (LLaVA)** | ⚠️ In Corpus (Context Only) | 0 Qs (Unassessed) | 17x | `Visual Instruction Tuning.pdf` |

## 4. Performance Breakdown by Evaluated Paper

| Paper | Questions | Top-5 Doc Hit | Top-5 Page Hit (±1) | Completed LLM Ans | Avg Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BLIP-2** | 13 | 100.0% (13/13) | 76.9% (10/13) | 84.6% (11/13) | 0.7509 |
| **CLIP** | 12 | 100.0% (12/12) | 75.0% (9/12) | 58.3% (7/12) | 0.7598 |
| **Flamingo** | 12 | 91.7% (11/12) | 33.3% (4/12) | 91.7% (11/12) | 0.7261 |
| **GLIP** | 12 | 100.0% (12/12) | 58.3% (7/12) | 75.0% (9/12) | 0.7185 |
| **GroundCount** | 12 | 100.0% (12/12) | 83.3% (10/12) | 91.7% (11/12) | 0.7326 |
| **Grounding DINO** | 12 | 100.0% (12/12) | 66.7% (8/12) | 83.3% (10/12) | 0.7849 |
| **KOSMOS-2** | 12 | 91.7% (11/12) | 66.7% (8/12) | 75.0% (9/12) | 0.6807 |
| **PaLM-E** | 12 | 100.0% (12/12) | 75.0% (9/12) | 66.7% (8/12) | 0.7555 |
| **RT-2** | 12 | 100.0% (12/12) | 75.0% (9/12) | 91.7% (11/12) | 0.7306 |
| **Thinking with Visual Grounding** | 13 | 100.0% (13/13) | 100.0% (13/13) | 100.0% (13/13) | 0.7004 |

## 5. Performance Breakdown by Question Type

| Question Type | Questions | Top-5 Doc Hit | Top-5 Page Hit (±1) | Completed LLM Ans | Avg Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Architecture** | 12 | 100.0% (12/12) | 75.0% (9/12) | 100.0% (12/12) | 0.7700 |
| **Comparison** | 12 | 100.0% (12/12) | 66.7% (8/12) | 83.3% (10/12) | 0.7414 |
| **Definition** | 18 | 94.4% (17/18) | 55.6% (10/18) | 77.8% (14/18) | 0.6968 |
| **Numbers** | 11 | 100.0% (11/11) | 100.0% (11/11) | 54.5% (6/11) | 0.7126 |
| **Performance** | 13 | 100.0% (13/13) | 69.2% (9/13) | 69.2% (9/13) | 0.7781 |
| **Procedure** | 35 | 97.1% (34/35) | 74.3% (26/35) | 85.7% (30/35) | 0.7322 |
| **Reason/Why** | 17 | 100.0% (17/17) | 70.6% (12/17) | 94.1% (16/17) | 0.7309 |
| **Training** | 4 | 100.0% (4/4) | 50.0% (2/4) | 75.0% (3/4) | 0.7107 |

## 6. Complete Questions & Diagnostic Index

| ID | Paper | Type | Top-1 Score | Retrieval Match | Failure / Diagnostic | Question | Golden Answer | Generated Answer |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| `blip2_001` | BLIP-2 | Architecture | 0.811 | Exact Page (Rank 2) | ✅ Correct / Grounded | What core module does BLIP-2 introduce to bridge t... | The Querying Transformer (Q-Former). | The core module introduced by BLIP-2 to bridge the... |
| `blip2_002` | BLIP-2 | Procedure | 0.720 | Exact Page (Rank 5) | ✅ Correct / Grounded | What are the two pre-training stages in BLIP-2? | Stage 1 bootstraps vision-language repre... | Based on the provided context, the two pre-trainin... |
| `blip2_003` | BLIP-2 | Numbers | 0.692 | Exact Page (Rank 5) | A. Retrieval failure | How many total parameters does Q-Former contain, a... | 188M parameters, initialized with the pr... | Based on the provided context, there is no informa... |
| `blip2_004` | BLIP-2 | Numbers | 0.698 | Exact Page (Rank 4) | B. Context contains answer, but LLM failed | How many learnable query embeddings are used in Q-... | 32 queries, each having a dimension of 7... | Based on the provided context, **32** learnable qu... |
| `blip2_005` | BLIP-2 | Training | 0.740 | Exact Page (Rank 4) | ✅ Correct / Grounded | What three pre-training objectives are jointly opt... | Image-Text Contrastive Learning (ITC), I... | Based on the provided context, the three pre-train... |
| `blip2_006` | BLIP-2 | Architecture | 0.738 | Exact Page (Rank 1) | ✅ Correct / Grounded | What self-attention masking strategies are applied... | A unimodal self-attention mask for ITC, ... | Based on the provided context, the self-attention ... |
| `blip2_007` | BLIP-2 | Procedure | 0.726 | Doc Only (Rank 1) | A. Retrieval failure | Which specific layer of the frozen Vision Transfor... | The second-to-last layer (the last layer... | Based on the provided context, the specific layer ... |
| `blip2_008` | BLIP-2 | Architecture | 0.799 | Exact Page (Rank 1) | ✅ Correct / Grounded | How is the output of the Q-Former adapted to the i... | Via a fully-connected (FC) layer that li... | In the generative stage, a fully-connected (FC) la... |
| `blip2_009` | BLIP-2 | Comparison | 0.763 | Doc Only (Rank 1) | ✅ Correct / Grounded | How does BLIP-2 compare to Flamingo80B on zero-sho... | BLIP-2 outperforms Flamingo80B by 8.7% o... | Based on the provided context, BLIP-2 outperforms ... |
| `blip2_010` | BLIP-2 | Reason/Why | 0.730 | Exact Page (Rank 1) | ✅ Correct / Grounded | What happens if vision-to-language generative lear... | The model gives substantially lower zero... | If vision-to-language generative learning is train... |
| `blip2_011` | BLIP-2 | Reason/Why | 0.742 | Doc Only (Rank 5) | C. LLM hallucination | Why does BLIP-2 lack in-context few-shot learning ... | Because its pre-training dataset contain... | Based on the provided context, **BLIP-2 lacks in-c... |
| `blip2_012` | BLIP-2 | Architecture | 0.872 | Exact Page (Rank 1) | ✅ Correct / Grounded | How is the question provided to Q-Former during vi... | Question tokens are provided as input to... | During VQA fine-tuning, the question tokens are gi... |
| `blip2_013` | BLIP-2 | Performance | 0.732 | Exact Page (Rank 1) | B. Context contains answer, but LLM failed | What score did fine-tuned BLIP-2 ViT-G achieve on ... | 121.6 overall CIDEr (with FlanT5_XL) and... | The provided context does not contain information ... |
| `flam_001` | Flamingo | Architecture | 0.734 | Adjacent Page (Rank 2) | B. Context contains answer, but LLM failed | What vision encoder architecture does Flamingo use... | A pretrained and frozen Normalizer-Free ... | Based on the provided context, Flamingo uses a **P... |
| `flam_002` | Flamingo | Definition | 0.691 | Doc Only (Rank 1) | ✅ Correct / Grounded | What is the function of the Perceiver Resampler in... | It takes variable-size spatio-temporal f... | Based on the provided context, the function of the... |
| `flam_003` | Flamingo | Architecture | 0.732 | Doc Only (Rank 2) | ✅ Correct / Grounded | What mechanism ensures that the language model out... | A tanh-gating mechanism where the newly ... | Based on the provided context, the mechanism that ... |
| `flam_004` | Flamingo | Numbers | 0.763 | Adjacent Page (Rank 1) | ✅ Correct / Grounded | How frequently are GATED XATTN-DENSE layers insert... | Every fourth block for Flamingo-9B, and ... | Based on the provided context, GATED XATTN-DENSE l... |
| `flam_005` | Flamingo | Procedure | 0.777 | Exact Page (Rank 1) | ✅ Correct / Grounded | How does Flamingo's per-image cross-attention mask... | At any text token, the model only cross-... | Based on the provided context, Flamingo's per-imag... |
| `flam_006` | Flamingo | Training | 0.669 | Doc Only (Rank 1) | ✅ Correct / Grounded | What three datasets comprise Flamingo's web-scrape... | M3W (MultiModal Massive Web interleaved ... | Based on the provided context, there are not exact... |
| `flam_007` | Flamingo | Comparison | 0.653 | Doc Only (Rank 1) | B. Context contains answer, but LLM failed | How does gradient accumulation compare to round-ro... | Accumulating and weighting gradients ove... | Based on the provided context, **gradient accumula... |
| `flam_008` | Flamingo | Reason/Why | 0.738 | Doc Only (Rank 1) | B. Context contains answer, but LLM failed | Why is the pretrained language model kept frozen d... | To prevent catastrophic forgetting, whic... | Based on the provided context, the pretrained lang... |
| `flam_009` | Flamingo | Performance | 0.790 | Exact Page (Rank 1) | B. Context contains answer, but LLM failed | On how many tasks did Flamingo-80B with 32 shots o... | On 6 of the 16 evaluated benchmarks. | Based on Chunk 1, Flamingo-80B with 32 shots outpe... |
| `flam_010` | Flamingo | Procedure | 0.768 | Doc Only (Rank 4) | B. Context contains answer, but LLM failed | How does Flamingo construct zero-shot prompts for ... | By providing two text-only examples from... | Based on the provided context, Flamingo constructs... |
| `flam_011` | Flamingo | Procedure | 0.649 | Doc Miss | A. Retrieval failure | What is RICES, and how does it order support examp... | Retrieval-based In-Context Example Selec... | Based on the provided context, **RICES** is not ex... |
| `flam_012` | Flamingo | Reason/Why | 0.748 | Doc Only (Rank 1) | ✅ Correct / Grounded | Why does Flamingo underperform contrastive models ... | Contrastive models directly optimize for... | Based on the provided context, Flamingo underperfo... |
| `glip_001` | GLIP | Procedure | 0.842 | Exact Page (Rank 5) | ✅ Correct / Grounded | How does GLIP reformulate object detection as a ph... | By converting candidate object categorie... | Based on the provided context, GLIP reformulates o... |
| `glip_002` | GLIP | Definition | 0.708 | Doc Only (Rank 1) | ✅ Correct / Grounded | What is deep cross-modality fusion in GLIP, and wh... | Cross-modality multi-head attention (X-M... | Based on the provided context, **deep cross-modali... |
| `glip_003` | GLIP | Procedure | 0.758 | Exact Page (Rank 4) | ✅ Correct / Grounded | How does GLIP scale up grounding data using web-cr... | A pre-trained teacher GLIP detects bound... | GLIP scales up grounding data by automatically gen... |
| `glip_004` | GLIP | Performance | 0.804 | Doc Only (Rank 1) | B. Context contains answer, but LLM failed | What zero-shot AP did GLIP-L achieve on COCO 2017v... | 49.8 AP on COCO val2017 and 26.9 AP on L... | Based on the provided context, there is no informa... |
| `glip_005` | GLIP | Performance | 0.746 | Doc Only (Rank 1) | C. LLM hallucination | What AP did GLIP-L achieve after full fine-tuning ... | 60.8 AP on COCO 2017val and 61.5 AP on t... | 69.6 |
| `glip_006` | GLIP | Training | 0.730 | Exact Page (Rank 2) | A. Retrieval failure | What datasets comprise GoldG in GLIP's pre-trainin... | 0.8M human-annotated gold grounding data... | Based on the provided context, there is no mention... |
| `glip_007` | GLIP | Definition | 0.665 | Doc Only (Rank 2) | A. Retrieval failure | What is ODinW in the context of GLIP's evaluation? | Object Detection in the Wild: a benchmar... | Based on the provided context, **ODinW** is not ex... |
| `glip_008` | GLIP | Comparison | 0.754 | Doc Only (Rank 1) | ✅ Correct / Grounded | How does 1-shot GLIP-L compare to Dynamic Head on ... | A 1-shot GLIP-L rivals a fully supervise... | Based on the provided context, a **1-shot GLIP-L**... |
| `glip_009` | GLIP | Procedure | 0.690 | Exact Page (Rank 2) | ✅ Correct / Grounded | What is manual prompt tuning in GLIP, and what is ... | Expanding prompt category names with vis... | Based on the provided context, **manual prompt tun... |
| `glip_010` | GLIP | Reason/Why | 0.710 | Adjacent Page (Rank 3) | ✅ Correct / Grounded | Why is prompt tuning effective for GLIP while line... | Because GLIP's language-aware deep fusio... | Based on the provided context, prompt tuning is ef... |
| `glip_011` | GLIP | Procedure | 0.724 | Exact Page (Rank 1) | C. LLM hallucination | How does GLIP handle vocabulary prompts that excee... | It keeps all positive classes present in... | When object categories cannot fit into a single pr... |
| `glip_012` | GLIP | Comparison | 0.493 | Adjacent Page (Rank 4) | B. Context contains answer, but LLM failed | What did the ablation comparing 2.66M detection da... | FourODs achieved only 15.0 APr, whereas ... | The provided context does not contain information ... |
| `gdino_001` | Grounding DINO | Architecture | 0.809 | Doc Only (Rank 3) | C. LLM hallucination | What base object detector does Grounding DINO buil... | DINO (DETR with improved denoising ancho... | Grounding DINO builds upon **DINO**. This is expli... |
| `gdino_002` | Grounding DINO | Architecture | 0.764 | Doc Only (Rank 1) | ✅ Correct / Grounded | What are the three stages where Grounding DINO per... | Phase A: Neck (Feature Enhancer), Phase ... | Based on Chunk 2 (Page 6), the three stages/module... |
| `gdino_003` | Grounding DINO | Procedure | 0.778 | Exact Page (Rank 1) | ✅ Correct / Grounded | How does Grounding DINO's Language-Guided Query Se... | It computes image-text feature similarit... | Grounding DINO's Language-Guided Query Selection m... |
| `gdino_004` | Grounding DINO | Definition | 0.815 | Adjacent Page (Rank 1) | A. Retrieval failure | What is 'sub-sentence level text feature represent... | An attention masking strategy that block... | Based on the provided context, the **sub-sentence ... |
| `gdino_005` | Grounding DINO | Performance | 0.819 | Exact Page (Rank 5) | C. LLM hallucination | What zero-shot AP did Grounding DINO L achieve on ... | 52.5 AP on COCO 2017val. | 63.0 |
| `gdino_006` | Grounding DINO | Performance | 0.820 | Doc Only (Rank 1) | B. Context contains answer, but LLM failed | What zero-shot mean AP did Grounding DINO L achiev... | 26.1 mean AP. | 26.1 |
| `gdino_007` | Grounding DINO | Numbers | 0.759 | Exact Page (Rank 1) | A. Retrieval failure | How many queries and feature enhancer layers does ... | 900 queries and 6 feature enhancer layer... | Based on the provided context, the specific number... |
| `gdino_008` | Grounding DINO | Procedure | 0.766 | Adjacent Page (Rank 2) | C. LLM hallucination | How is Referring Expression Comprehension (REC) ev... | The text expression is fed as the text p... | Based on the provided context, Referring Expressio... |
| `gdino_009` | Grounding DINO | Reason/Why | 0.757 | Adjacent Page (Rank 1) | D. Question ambiguity/difficulty | Why do Grounding DINO and GLIP achieve poor zero-s... | Because grounded object detection models... | Based on the provided context, Grounding DINO and ... |
| `gdino_010` | Grounding DINO | Procedure | 0.827 | Exact Page (Rank 1) | ✅ Correct / Grounded | Can a pre-trained DINO detector be converted into ... | Yes, by freezing the pre-existing DINO m... | Yes, a pre-trained DINO detector can be converted ... |
| `gdino_011` | Grounding DINO | Procedure | 0.788 | Exact Page (Rank 2) | ✅ Correct / Grounded | How is Grounding DINO integrated with Stable Diffu... | Grounding DINO detects target objects to... | Grounding DINO is integrated with Stable Diffusion... |
| `gdino_012` | Grounding DINO | Comparison | 0.715 | Doc Only (Rank 1) | A. Retrieval failure | How does Grounding DINO T compare to GLIP-T in ter... | Grounding DINO T has 172M parameters and... | Based on the provided context, there is no informa... |
| `gcount_001` | GroundCount | Reason/Why | 0.884 | Exact Page (Rank 2) | ✅ Correct / Grounded | Why do state-of-the-art reasoning VLMs still exhib... | Due to fundamental spatial-semantic inte... | According to the provided context, state-of-the-ar... |
| `gcount_002` | GroundCount | Definition | 0.678 | Exact Page (Rank 1) | ✅ Correct / Grounded | What is GroundCount A? | A prompt-based augmentation strategy whe... | GroundCount A is a prompt-based ODM (Object Detect... |
| `gcount_003` | GroundCount | Procedure | 0.760 | Exact Page (Rank 1) | ✅ Correct / Grounded | How does GroundCount A encode spatial locations in... | It discretizes the image into a 3x3 grid... | Based on the provided context, **GroundCount A** e... |
| `gcount_004` | GroundCount | Procedure | 0.786 | Adjacent Page (Rank 1) | D. Question ambiguity/difficulty | In what order are detected objects sequenced in th... | First by horizontal position (left-to-ri... | Based on the provided context, the detected object... |
| `gcount_005` | GroundCount | Definition | 0.714 | Exact Page (Rank 5) | ✅ Correct / Grounded | What is the exact string template format used for ... | '[class] [index] [position]: [confidence... | Based on Chunk 1, the exact string template format... |
| `gcount_006` | GroundCount | Performance | 0.804 | Exact Page (Rank 5) | ✅ Correct / Grounded | What counting accuracy improvement and latency red... | It improved counting accuracy by 6.6 per... | GroundCount A achieved a **6.6 percentage point (p... |
| `gcount_007` | GroundCount | Reason/Why | 0.737 | Doc Only (Rank 1) | D. Question ambiguity/difficulty | Why does prompt augmentation reduce total VLM infe... | By eliminating hallucination-driven reas... | Based on the provided context, prompt augmentation... |
| `gcount_008` | GroundCount | Comparison | 0.798 | Exact Page (Rank 4) | ✅ Correct / Grounded | How did GroundCount A (prompt augmentation) compar... | Explicit prompt augmentation (81.3%) out... | Based on the provided context, **GroundCount A** (... |
| `gcount_009` | GroundCount | Procedure | 0.582 | Doc Only (Rank 1) | A. Retrieval failure | What were the ablation findings regarding includin... | Confidence scores introduced noise; remo... | Based on the provided context, there are no specif... |
| `gcount_010` | GroundCount | Comparison | 0.745 | Exact Page (Rank 1) | ✅ Correct / Grounded | How did removing positional encoding impact strong... | It degraded stronger models (Molmo2-4B, ... | Removing positional encoding caused degradation in... |
| `gcount_011` | GroundCount | Procedure | 0.665 | Exact Page (Rank 3) | ✅ Correct / Grounded | What happens when the ODM detection confidence thr... | Performance degrades uniformly across al... | Lowering the confidence threshold from 0.5 to 0.3 ... |
| `gcount_012` | GroundCount | Reason/Why | 0.640 | Exact Page (Rank 1) | ✅ Correct / Grounded | Why did InternVL3.5-1B degrade under full ODM prom... | Its iterative reflection mechanisms were... | Based on the provided context, **InternVL3.5-1B** ... |
| `kos_001` | KOSMOS-2 | Definition | 0.566 | Doc Only (Rank 1) | ✅ Correct / Grounded | What two capabilities does KOSMOS-2 introduce comp... | Multimodal grounding (e.g., phrase groun... | KOSMOS-2 introduces new capabilities of **groundin... |
| `kos_002` | KOSMOS-2 | Architecture | 0.782 | Exact Page (Rank 1) | ✅ Correct / Grounded | How does KOSMOS-2 represent referring expressions ... | As Markdown-style hyperlinks using bound... | KOSMOS-2 represents refer expressions as links in ... |
| `kos_003` | KOSMOS-2 | Numbers | 0.587 | Exact Page (Rank 2) | ✅ Correct / Grounded | How are continuous 2D coordinates discretized into... | The image is evenly divided into 32x32 b... | Based on the provided context, continuous 2D coord... |
| `kos_004` | KOSMOS-2 | Numbers | 0.570 | Exact Page (Rank 2) | A. Retrieval failure | What are the dataset statistics of GRIT constructe... | Approximately 91M images, 115M text span... | Based on the provided context, there is no informa... |
| `kos_005` | KOSMOS-2 | Procedure | 0.583 | Doc Only (Rank 1) | A. Retrieval failure | What are the two steps in the GRIT data constructi... | Step 1: generating noun-chunk-bounding-b... | Based on the provided context, there is no informa... |
| `kos_006` | KOSMOS-2 | Procedure | 0.544 | Doc Only (Rank 1) | C. LLM hallucination | What filtering is applied in Step 1 of GRIT data c... | Abstract nouns (e.g., 'time', 'love') ar... | Based on the provided context, **no specific filte... |
| `kos_007` | KOSMOS-2 | Definition | 0.755 | Doc Miss | A. Retrieval failure | What special token prompts KOSMOS-2 to ground its ... | The '<grounding>' token. | Based on the provided context, there is no informa... |
| `kos_008` | KOSMOS-2 | Architecture | 0.664 | Exact Page (Rank 2) | ✅ Correct / Grounded | What model architecture and parameter count are us... | A 24-layer MAGNETO Transformer causal la... | Based on the provided context, the KOSMOS-2 model ... |
| `kos_009` | KOSMOS-2 | Performance | 0.759 | Exact Page (Rank 5) | ✅ Correct / Grounded | What zero-shot Recall@1 did KOSMOS-2 achieve on Fl... | 77.8% on the validation split and 78.7% ... | Based on Table 2 in Chunk 5, KOSMOS-2 achieved a *... |
| `kos_010` | KOSMOS-2 | Comparison | 0.861 | Exact Page (Rank 1) | ✅ Correct / Grounded | How did zero-shot KOSMOS-2 compare to the fine-tun... | Zero-shot KOSMOS-2 scored 60.3 CIDEr, ou... | Zero-shot KOSMOS-2 outperformed the finetuned SLR ... |
| `kos_011` | KOSMOS-2 | Procedure | 0.768 | Exact Page (Rank 1) | A. Retrieval failure | What prompt template is used to evaluate referring... | '<p> It </p><box><loc_1><loc_2></box> is... | Based on the provided context, the specific prompt... |
| `kos_012` | KOSMOS-2 | Reason/Why | 0.729 | Exact Page (Rank 2) | ✅ Correct / Grounded | Why does KOSMOS-2 perform better on RefCOCOg than ... | RefCOCO and RefCOCO+ use short, game-bas... | KOSMOS-2 performs better on RefCOCOg than on RefCO... |
| `clip_001` | CLIP | Training | 0.704 | Doc Only (Rank 1) | A. Retrieval failure | What is CLIP's core pre-training objective? | Predicting correct (image, text) pairing... | The provided context does not contain information ... |
| `clip_002` | CLIP | Numbers | 0.729 | Adjacent Page (Rank 2) | C. LLM hallucination | What is the name and scale of the pre-training dat... | WebImageText (WIT), containing 400 milli... | The pre-training dataset created for CLIP is named... |
| `clip_003` | CLIP | Reason/Why | 0.730 | Exact Page (Rank 1) | B. Context contains answer, but LLM failed | Why did CLIP adopt a contrastive objective instead... | The contrastive objective was 4x more co... | The provided text does not contain information reg... |
| `clip_004` | CLIP | Procedure | 0.794 | Doc Only (Rank 1) | B. Context contains answer, but LLM failed | How does CLIP synthesize a zero-shot linear classi... | The text encoder embeds class names or p... | Based on the provided context, CLIP synthesizes a ... |
| `clip_005` | CLIP | Performance | 0.782 | Doc Only (Rank 1) | A. Retrieval failure | What zero-shot Top-1 accuracy did CLIP ViT-L/14@33... | 76.2% Top-1 accuracy, matching a fully s... | The provided context does not contain information ... |
| `clip_006` | CLIP | Procedure | 0.824 | Adjacent Page (Rank 1) | B. Context contains answer, but LLM failed | How much does prompt engineering and ensembling im... | By almost 5 percentage points on average... | Based on the provided context, prompt engineering ... |
| `clip_007` | CLIP | Definition | 0.663 | Adjacent Page (Rank 4) | A. Retrieval failure | What is the difference between effective robustnes... | Effective robustness measures out-of-dis... | Based on the provided context, there is no explici... |
| `clip_008` | CLIP | Performance | 0.862 | Exact Page (Rank 4) | B. Context contains answer, but LLM failed | By how much does zero-shot CLIP reduce the robustn... | By up to 75%. | Zero-shot CLIP reduces the "robustness gap" by up ... |
| `clip_009` | CLIP | Reason/Why | 0.772 | Adjacent Page (Rank 3) | B. Context contains answer, but LLM failed | What happens to CLIP's effective robustness when f... | ImageNet accuracy increases by 9.2%, but... | Based on the provided context, when fine-tuned on ... |
| `clip_010` | CLIP | Numbers | 0.725 | Exact Page (Rank 2) | B. Context contains answer, but LLM failed | What mini-batch size and temperature parameter ini... | A mini-batch size of 32,768, with temper... | The provided text does not contain information reg... |
| `clip_011` | CLIP | Comparison | 0.825 | Exact Page (Rank 3) | ✅ Correct / Grounded | On which tasks does zero-shot CLIP perform poorly ... | Fine-grained classification (aircraft, f... | Based on the provided context, zero-shot CLIP perf... |
| `clip_012` | CLIP | Procedure | 0.708 | Exact Page (Rank 1) | ✅ Correct / Grounded | What did CLIP's data overlap and contamination ana... | Across 35 datasets, median overlap was 2... | Based on the provided context, CLIP's data overlap... |
| `palme_001` | PaLM-E | Definition | 0.786 | Adjacent Page (Rank 2) | ✅ Correct / Grounded | What is the core architectural concept of PaLM-E? | Injecting continuous embodied sensor obs... | Based on the provided context, the core architectu... |
| `palme_002` | PaLM-E | Numbers | 0.743 | Adjacent Page (Rank 2) | A. Retrieval failure | What are the parameter counts of the vision and la... | A 22B parameter Vision Transformer (ViT-... | Based on the provided context, specifically Chunk ... |
| `palme_003` | PaLM-E | Definition | 0.646 | Doc Only (Rank 1) | C. LLM hallucination | What is OSRT in PaLM-E, and what advantage did it ... | Object Scene Representation Transformer;... | Based on the provided context, **OSRT** stands for... |
| `palme_004` | PaLM-E | Procedure | 0.740 | Doc Only (Rank 1) | A. Retrieval failure | How does PaLM-E resolve ambiguous object reference... | Via entity referrals: labeling object to... | Based on the provided context, there is no informa... |
| `palme_005` | PaLM-E | Procedure | 0.736 | Adjacent Page (Rank 4) | ✅ Correct / Grounded | How does PaLM-E operate within a closed-loop robot... | As a high-level policy that autoregressi... | PaLM-E operates within a closed-loop robot control... |
| `palme_006` | PaLM-E | Numbers | 0.828 | Adjacent Page (Rank 3) | B. Context contains answer, but LLM failed | What percentage of the full training mixture datas... | Less than 10% (specifically 8.9% across ... | Based on the provided context, there are two diffe... |
| `palme_007` | PaLM-E | Definition | 0.770 | Exact Page (Rank 3) | C. LLM hallucination | What is 'positive transfer' in PaLM-E? | Co-training on internet-scale vision-lan... | Based on the provided context, **positive transfer... |
| `palme_008` | PaLM-E | Performance | 0.712 | Exact Page (Rank 3) | A. Retrieval failure | What accuracy did PaLM-E-562B achieve on the OK-VQ... | 66.1%, achieving state-of-the-art perfor... | The provided context does not contain information ... |
| `palme_009` | PaLM-E | Reason/Why | 0.819 | Doc Only (Rank 1) | ✅ Correct / Grounded | How does model scale affect catastrophic forgettin... | Larger models experience significantly l... | Based on the provided context, increasing the mode... |
| `palme_010` | PaLM-E | Performance | 0.695 | Adjacent Page (Rank 1) | B. Context contains answer, but LLM failed | What emergent capability was observed in PaLM-E-56... | Zero-shot multi-image reasoning and ques... | The ability to reason over multiple images. |
| `palme_011` | PaLM-E | Procedure | 0.800 | Adjacent Page (Rank 3) | ✅ Correct / Grounded | What three robot manipulation domains are evaluate... | A simulated Task and Motion Planning (TA... | Based on the provided context, the three robot man... |
| `palme_012` | PaLM-E | Performance | 0.790 | Adjacent Page (Rank 4) | ✅ Correct / Grounded | What F1 scores did PaLM-E-12B achieve on affordanc... | 0.91 F1 on affordance prediction and 0.9... | Based on the provided context, here are the F1 sco... |
| `rt2_001` | RT-2 | Definition | 0.818 | Doc Only (Rank 1) | ✅ Correct / Grounded | What is a Vision-Language-Action (VLA) model in RT... | A model fine-tuned from a pre-trained vi... | Based on the provided context, a **Vision-Language... |
| `rt2_002` | RT-2 | Architecture | 0.718 | Adjacent Page (Rank 4) | C. LLM hallucination | What robot action space is used in RT-2, and how i... | An 8-dimensional action space (3D displa... | Based on the provided context, the specific detail... |
| `rt2_003` | RT-2 | Procedure | 0.680 | Exact Page (Rank 1) | A. Retrieval failure | How are action tokens mapped in RT-2-PaLI-X versus... | RT-2-PaLI-X uses integer tokens up to 10... | Based on the provided context, there is no informa... |
| `rt2_004` | RT-2 | Definition | 0.512 | Doc Only (Rank 2) | C. LLM hallucination | What is 'co-fine-tuning' in RT-2, and why is it im... | Jointly fine-tuning on both robot trajec... | Based on the provided context, **'co-fine-tuning'*... |
| `rt2_005` | RT-2 | Procedure | 0.766 | Exact Page (Rank 1) | ✅ Correct / Grounded | How does RT-2 achieve real-time inference on a phy... | By deploying the model on a multi-TPU cl... | RT-2 achieves real-time inference for its 55B para... |
| `rt2_006` | RT-2 | Comparison | 0.827 | Exact Page (Rank 3) | ✅ Correct / Grounded | How does RT-2 compare to RT-1 on seen tasks versus... | Performance on seen tasks is comparable ... | Based on the provided context, here is how RT-2 co... |
| `rt2_007` | RT-2 | Definition | 0.721 | Exact Page (Rank 3) | ✅ Correct / Grounded | What are the three categories of emergent capabili... | Symbol understanding, reasoning (visual,... | Based on the provided context, the three categorie... |
| `rt2_008` | RT-2 | Reason/Why | 0.761 | Exact Page (Rank 1) | ✅ Correct / Grounded | Why does RT-2-PaLM-E outperform RT-2-PaLI-X on mat... | Because PaLM-E's pre-training mixture in... | Based on the provided context, **RT-2-PaLM-E** out... |
| `rt2_009` | RT-2 | Procedure | 0.785 | Doc Only (Rank 1) | ✅ Correct / Grounded | How is Chain-of-Thought (CoT) reasoning implemente... | By training the model to predict a natur... | Based on the provided context, Chain-of-Thought (C... |
| `rt2_010` | RT-2 | Reason/Why | 0.660 | Exact Page (Rank 1) | ✅ Correct / Grounded | What fundamental physical limitation remains in RT... | It cannot execute new physical motions b... | Despite web-scale pretraining, the robot does not ... |
| `rt2_011` | RT-2 | Numbers | 0.746 | Exact Page (Rank 1) | B. Context contains answer, but LLM failed | What data sampling ratios were used for the roboti... | About 50% for RT-2-PaLI-X and about 66% ... | Based on the provided context, the specific numeri... |
| `rt2_012` | RT-2 | Procedure | 0.774 | Exact Page (Rank 1) | ✅ Correct / Grounded | How does RT-2 enforce valid action output generati... | By constraining output decoding to sampl... | RT-2 enforces valid action output generation by co... |
| `twvg_001` | Thinking with Visual Grounding | Definition | 0.714 | Exact Page (Rank 1) | ✅ Correct / Grounded | What is 'visually grounded thinking'? | A visual reasoning process in which mode... | Visually grounded thinking is a reasoning process ... |
| `twvg_002` | Thinking with Visual Grounding | Architecture | 0.816 | Exact Page (Rank 1) | ✅ Correct / Grounded | What base vision-language model is used across the... | Gemma3-4B-IT. | Gemma3-4B-IT |
| `twvg_003` | Thinking with Visual Grounding | Definition | 0.636 | Adjacent Page (Rank 3) | ✅ Correct / Grounded | What tag formatting is used for box grounding and ... | Box mode: '<obj> name phrase \| [x1, y1,... | Based on the provided context, the tag formatting ... |
| `twvg_004` | Thinking with Visual Grounding | Procedure | 0.730 | Adjacent Page (Rank 1) | ✅ Correct / Grounded | How are visual grounding masks acquired in the dat... | Using an iterative VLM agent calling SAM... | Visual grounding masks are acquired through a **SA... |
| `twvg_005` | Thinking with Visual Grounding | Procedure | 0.668 | Exact Page (Rank 2) | C. LLM hallucination | How does the data synthesis pipeline derive single... | By choosing the interior point farthest ... | Based on the provided context, the data synthesis ... |
| `twvg_006` | Thinking with Visual Grounding | Definition | 0.685 | Exact Page (Rank 1) | ✅ Correct / Grounded | What is the function of the 'grounding object rout... | A lightweight VLM (Qwen3.5-4B) that matc... | The function of the 'grounding object router' duri... |
| `twvg_007` | Thinking with Visual Grounding | Procedure | 0.664 | Exact Page (Rank 2) | ✅ Correct / Grounded | How is box grounding quality calculated in the rew... | By computing the Intersection-over-Union... | Based on the provided context, box grounding quali... |
| `twvg_008` | Thinking with Visual Grounding | Procedure | 0.718 | Exact Page (Rank 1) | ✅ Correct / Grounded | How is point grounding quality calculated in the r... | By computing the per-object F1 score bas... | Based on the provided context, point grounding qua... |
| `twvg_009` | Thinking with Visual Grounding | Reason/Why | 0.631 | Exact Page (Rank 3) | B. Context contains answer, but LLM failed | Why are unmatched grounding objects in the model r... | Because the model may identify and groun... | Unmatched grounding objects in the model rollout a... |
| `twvg_010` | Thinking with Visual Grounding | Procedure | 0.740 | Adjacent Page (Rank 4) | A. Retrieval failure | What reinforcement learning algorithm is used to t... | GRPO (Group Relative Policy Optimization... | Based on the provided context, the specific name o... |
| `twvg_011` | Thinking with Visual Grounding | Reason/Why | 0.638 | Exact Page (Rank 1) | ✅ Correct / Grounded | What failure mode did the non-grounded thinking ba... | Length collapse, where the generated rea... | The non-grounded thinking baseline suffered from *... |
| `twvg_012` | Thinking with Visual Grounding | Comparison | 0.664 | Exact Page (Rank 2) | ✅ Correct / Grounded | Which grounding interface (points vs. boxes) perfo... | Point grounding performs better, because... | Based on the provided context, **point-mode ground... |
| `twvg_013` | Thinking with Visual Grounding | Comparison | 0.800 | Exact Page (Rank 5) | ✅ Correct / Grounded | How do the 4B visually grounded thinking models co... | They achieve comparable performance over... | On spatial reasoning benchmarks, the 4B visually g... |

---
*Interactive Dashboard available at `evaluation/results/report.html`*

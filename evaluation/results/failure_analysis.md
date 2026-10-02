# 🔍 RAG Evaluation Failure Taxonomy & Error Analysis

**Baseline Dataset**: `experiment_0_baseline.json` (122 Benchmark Questions)

**Evaluation Scope**: 10 Tested Research Papers · Golden Benchmark Suite

---

## 1. Executive Failure Taxonomy Breakdown

| Failure Category | Count | % of All Qs | % of Failures | Core Diagnostic Description & Remedy |
| :--- | :---: | :---: | :---: | :--- |
| **A. Retrieval failure** | **21** | **17.2%** | **36.8%** | Increase Top-K (5 -> 10), add BM25 Hybrid Search, test Markdown/Recursive chunking to capture missing pages. |
| **B. Context contains answer, but LLM failed** | **20** | **16.4%** | **35.1%** | Tune system prompt to guide table extraction; highlight source chunks; test larger extraction context window. |
| **C. LLM hallucination** | **13** | **10.7%** | **22.8%** | Enforce strict refusal prompt ('If exact fact is missing, refuse. Do not guess acronyms or extrapolate numbers'). |
| **D. Question ambiguity/difficulty** | **3** | **2.5%** | **5.3%** | Refine question definitions in golden dataset or provide multi-hop retriever to synthesize across distant sections. |
| **Passing / Accurate Answers** | **65** | **53.3%** | — | Fully grounded and factually aligned with golden answer |

## 2. Failure Distribution by Research Paper

| Research Paper | Total Qs | Passing | Cat A (Retrieval) | Cat B (LLM Extract) | Cat C (Hallucination) | Cat D (Ambiguity) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BLIP-2** | 13 | 8 (61.5%) | 2 | 2 | 1 | 0 |
| **CLIP** | 12 | 2 (16.7%) | 3 | 6 | 1 | 0 |
| **Flamingo** | 12 | 6 (50.0%) | 1 | 5 | 0 | 0 |
| **GLIP** | 12 | 6 (50.0%) | 2 | 2 | 2 | 0 |
| **GroundCount** | 12 | 9 (75.0%) | 1 | 0 | 0 | 2 |
| **Grounding DINO** | 12 | 4 (33.3%) | 3 | 1 | 3 | 1 |
| **KOSMOS-2** | 12 | 7 (58.3%) | 4 | 0 | 1 | 0 |
| **PaLM-E** | 12 | 5 (41.7%) | 3 | 2 | 2 | 0 |
| **RT-2** | 12 | 8 (66.7%) | 1 | 1 | 2 | 0 |
| **Thinking with Visual Grounding** | 13 | 10 (76.9%) | 1 | 1 | 1 | 0 |

---

## 3. Detailed Failure Case Log (Grouped by Category)

### A. Retrieval failure (21 Questions)

> **Remedy Strategy**: Increase Top-K (5 -> 10), add BM25 Hybrid Search, test Markdown/Recursive chunking to capture missing pages.

#### `[blip2_003]` BLIP-2 — Exact Page (Rank 5)
- **Question**: *"How many total parameters does Q-Former contain, and how is it initialized?"*
- **Golden Answer**: `188M parameters, initialized with the pre-trained weights of BERT_base (with randomly initialized cross-attention layers).`
- **Generated Answer**: `Based on the provided context, there is no information regarding the **total number of parameters** that the Q-Former contains, nor is there any description of **how it is initialized**. The text discusses the model's architecture, training processes (such as finetuning with VQA data or ITG loss), and specific mechanisms like using in-batch negatives instead of momentum queues, but it does not specify parameter counts or initialization methods.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Exact Page (Rank 5)). LLM adhered to strict context constraints and faithfully refused.

#### `[blip2_007]` BLIP-2 — Doc Only (Rank 1)
- **Question**: *"Which specific layer of the frozen Vision Transformer is used as output visual features in BLIP-2?"*
- **Golden Answer**: `The second-to-last layer (the last layer is removed).`
- **Generated Answer**: `Based on the provided context, the specific layer of the frozen Vision Transformer used as output visual features in BLIP-2 is **the last layer**.

This is explicitly stated in Chunk 1: "BLIP-2 effectively leverages both frozen pre-trained image models and language models. We bridge the modality gap using a Q-Former pre-trained in two-stages... where it feeds the most useful visual feature for the LLM..." and further clarified in Chunk 2 regarding the Querying Transformer (QFormer): "QFormer is a lightweight transformer which employs a set of learnable query vectors to extract visual features from the frozen image encoder." While the text does not explicitly say "last layer" in the snippets provided, standard practice for such encoders (ViT-L/ViT-G) and the description of extracting features via query vectors implies taking the final representation. However, strictly adhering to the provided text which mentions "frozen pre-trained image models" and the QFormer extracting features from them without naming a specific layer number in the snippets:

The context states that the **Q-Former** extracts visual features from the **frozen image encoder**. It does not explicitly name the layer number (e.g., "last layer") in the provided chunks, but it describes the process of feeding the "most useful visual feature" from the frozen model to the LLM via the Q-Former.

*Correction based on strict reading of provided text:*
The provided text mentions: "BLIP-2 effectively leverages both frozen pre-trained image models... We bridge the modality gap using a Q-Former... where it feeds the most useful visual feature for the LLM". It also says in Chunk 2: "QFormer is a lightweight transformer which employs a set of learnable query vectors to extract visual features from the frozen image encoder."

The text **does not explicitly state** which specific layer (e.g., "last layer") is used in the provided chunks. However, it implies the feature comes from the **frozen image encoder** via the **Q-Former**.

If I must infer from general knowledge often associated with this paper (which the context summarizes), it is the last layer, but based *only* on the text provided:
The text says the Q-Former extracts features from the **frozen image encoder**.

Let's re-read carefully to see if "last layer" appears.
Chunk 1: "...feeds the most useful visual feature..."
Chunk 2: "...extract visual features from the frozen image encoder."
Chunk 3: Discusses ViT-L and ViT-G (the models), but not the specific layer index.

Since the prompt asks to answer using **only** the context, and the context does not explicitly name "Layer X" or "Last Layer", I should state what the text *does* say about the source of the features.

However, often in these tasks, if the information is missing from the snippet but implied by the model architecture described (ViT), one might look for clues. But strictly, the text says:
1. Use a **frozen image encoder** (e.g., ViT-L, ViT-G).
2. A **Q-Former** extracts features from it using query vectors.

The text does not specify the layer number. I will formulate the answer based on the explicit description of the source and mechanism found in the text.

Wait, looking at Chunk 1 again: "BLIP-2... feeds the most useful visual feature for the LLM".
Chunk 2: "QFormer... extract visual features from the frozen image encoder."

There is no mention of a specific layer number (like "last") in the provided text chunks. I will answer based on the available information regarding the source and the component responsible.

**Answer Formulation:**
The context states that BLIP-2 uses a **Querying Transformer (Q-Former)** to extract visual features from a **frozen image encoder**. The Q-Former employs a set of learnable query vectors to perform this extraction. The text does not explicitly specify the layer number (e.g., "last layer") in the provided chunks, but it confirms the use of the **frozen image encoder** as the source.

*Self-Correction*: If the user expects a specific answer like "last layer" because they know the paper, I must stick to the text. The text says "frozen pre-trained image models". It does not say "last layer".

Let's check if there is any hidden clue.
Chunk 1: "BLIP-2... feeds the most useful visual feature..."
Chunk 2: "...extract visual features from the frozen image encoder."

Conclusion: The provided text does not explicitly name the specific layer (e.g., "last layer"). It only identifies the **frozen image encoder** and the **Q-Former** as the mechanism. I will state this limitation or describe the process`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Doc Only (Rank 1)). LLM adhered to strict context constraints and faithfully refused.

#### `[flam_011]` Flamingo — Doc Miss
- **Question**: *"What is RICES, and how does it order support examples in the prompt?"*
- **Golden Answer**: `Retrieval-based In-Context Example Selection retrieves visually similar examples using the vision encoder and orders them by increasing similarity so the most similar appears closest to the query.`
- **Generated Answer**: `Based on the provided context, **RICES** is not explicitly defined or described as a specific entity, acronym, or method within the text. The available chunks discuss related topics such as few-shot in-context learning, prompt construction for ChatGPT/GPT-4, machine-generated instruction-following samples, reasoning processes, and fusion networks, but they do not contain information regarding "RICES."

Similarly, the context does not explain **how examples are ordered** within a prompt. While Chunk 1 mentions using few-shot in-context-learning where examples include an input sample's context and output sample's response to generate conversation responses, it does not specify any particular ordering strategy (e.g., ascending, descending, or specific grouping) for these examples.

Therefore, the provided text does not contain the information required to answer what RICES is or how it orders support examples in the prompt.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Doc Miss). LLM adhered to strict context constraints and faithfully refused.

#### `[glip_006]` GLIP — Exact Page (Rank 2)
- **Question**: *"What datasets comprise GoldG in GLIP's pre-training?"*
- **Golden Answer**: `0.8M human-annotated gold grounding data from Flickr30K, Visual Genome Caption, and GQA (with COCO images removed).`
- **Generated Answer**: `Based on the provided context, there is no mention of a dataset or model named "GoldG". The text refers to "GLIP-T (A)", "GLIP-T (B)", and "GLIP-T (C)" as models pre-trained with extra data, but it does not specify what those datasets are.

The only specific datasets mentioned in the context are:
*   **PascalVOC** (used for evaluation)
*   **Objects365** (referred to as O365, used for pre-training and evaluation)
*   **COCO** (used for evaluation)
*   Specific novel categories/datasets mentioned in Figure 4 and Table 1: **AerialDrone**, **Aquarium**, **Rabbits**, **EgoHands**, **Mushrooms**, **Packages**, **Raccoon**, **Shellfish**, **Vehicles**, **Pistols**, **Pothole**.

The context states that GLIP models are "pre-trained with extra data" but does not list the specific composition of that pre-training data (which might include Objects365 and potentially other large-scale datasets like ImageNet or LAION, though those are not explicitly named in the provided text).`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Exact Page (Rank 2)). LLM adhered to strict context constraints and faithfully refused.

#### `[glip_007]` GLIP — Doc Only (Rank 2)
- **Question**: *"What is ODinW in the context of GLIP's evaluation?"*
- **Golden Answer**: `Object Detection in the Wild: a benchmark of 13 public Roboflow datasets covering diverse scenarios like drone view, ego-centric images, and rare species.`
- **Generated Answer**: `Based on the provided context, **ODinW** is not explicitly defined as a specific acronym or framework within the text. However, it appears to be the name of a dataset suite containing **35 datasets** used for evaluating Grounding DINO (as seen in Table 12 caption: "Detailed results on 35 datasets in ODinW of Grounding DINO...").

The context does not provide further details on what "ODinW" stands for or its specific composition beyond listing the various datasets included within it.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Doc Only (Rank 2)). LLM adhered to strict context constraints and faithfully refused.

#### `[gdino_004]` Grounding DINO — Adjacent Page (Rank 1)
- **Question**: *"What is 'sub-sentence level text feature representation' in Grounding DINO?"*
- **Golden Answer**: `An attention masking strategy that blocks cross-attention between unrelated category names in the prompt while preserving per-word token features for fine-grained grounding.`
- **Generated Answer**: `Based on the provided context, the **sub-sentence level text feature representation** is a specific design proposal within Grounding DINO used to utilize detection data for text prompts in a "more reasonable way." It is part of the model's approach to better fuse cross-modality information and extend DINO to open-set object detection by allowing it to detect arbitrary objects given texts as queries.

The context notes that this representation was proposed alongside other modifications (such as the language-guided query selection module) to improve performance, though it does not provide a detailed mathematical definition or further elaboration on its internal structure beyond its purpose in fusing text and image features.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Adjacent Page (Rank 1)). LLM adhered to strict context constraints and faithfully refused.

#### `[gdino_007]` Grounding DINO — Exact Page (Rank 1)
- **Question**: *"How many queries and feature enhancer layers does Grounding DINO use by default?"*
- **Golden Answer**: `900 queries and 6 feature enhancer layers.`
- **Generated Answer**: `Based on the provided context, the specific number of queries and feature enhancer layers used by default is not explicitly stated in the text. The documents mention a "Feature Enhancer Layer" (Chunk 2) and show an architecture diagram with multiple decoder layers (Chunk 2), but they do not specify the exact count for either components.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Exact Page (Rank 1)). LLM adhered to strict context constraints and faithfully refused.

#### `[gdino_012]` Grounding DINO — Doc Only (Rank 1)
- **Question**: *"How does Grounding DINO T compare to GLIP-T in terms of parameters and throughput (FPS)?"*
- **Golden Answer**: `Grounding DINO T has 172M parameters and runs at 8.37 FPS, compared to GLIP-T's 232M parameters and 6.11 FPS.`
- **Generated Answer**: `Based on the provided context, there is no information regarding the number of **parameters** or **throughput (FPS)** for either Grounding DINO T or GLIP-T. The text focuses exclusively on performance metrics such as Average Score, Median Score, and mAP across various datasets under different training settings (linear probing, few-shot, full-shot) and prompt configurations.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Doc Only (Rank 1)). LLM adhered to strict context constraints and faithfully refused.

#### `[gcount_009]` GroundCount — Doc Only (Rank 1)
- **Question**: *"What were the ablation findings regarding including detection confidence scores in prompts?"*
- **Golden Answer**: `Confidence scores introduced noise; removing them improved performance by 0.1 to 4.8 percentage points in four of five evaluated VLMs.`
- **Generated Answer**: `Based on the provided context, there are no specific ablation findings reported regarding the inclusion of **detection confidence scores** in prompts.

The available text discusses the following related topics:
*   **ODM Augmentation:** Describes including position and confidence values for objects to improve accuracy and reduce hallucination (Chunk 1).
*   **Prompt Corruption:** Reports performance decreases when using "Incomplete Prompts" or "Corrupted Prompts," but does not explicitly isolate detection confidence scores as a variable in an ablation study (Chunk 4).
*   **Flamingo Ablation:** Details experimental setups and results for Flamingo-3B on multimodal benchmarks, focusing on resampler sizes and gradient steps, without mentioning detection confidence scores (Chunk 3).

Therefore, the provided context does not contain the answer to what the ablation findings were regarding detection confidence scores.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Doc Only (Rank 1)). LLM adhered to strict context constraints and faithfully refused.

#### `[kos_004]` KOSMOS-2 — Exact Page (Rank 2)
- **Question**: *"What are the dataset statistics of GRIT constructed for KOSMOS-2?"*
- **Golden Answer**: `Approximately 91M images, 115M text spans, and 137M associated bounding boxes.`
- **Generated Answer**: `Based on the provided context, there is no information regarding the specific dataset statistics (such as the number of samples, image resolution, or caption length) of the GRIT corpus. The text mentions that "Data samples of GRIT are shown in the Appendix" and describes GRIT as a "large-scale dataset," but it does not provide numerical statistics within the given chunks.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Exact Page (Rank 2)). LLM adhered to strict context constraints and faithfully refused.

#### `[kos_005]` KOSMOS-2 — Doc Only (Rank 1)
- **Question**: *"What are the two steps in the GRIT data construction pipeline?"*
- **Golden Answer**: `Step 1: generating noun-chunk-bounding-box pairs using spaCy and GLIP; Step 2: expanding noun chunks into referring-expression-bounding-box pairs via dependency trees.`
- **Generated Answer**: `Based on the provided context, there is no information describing a "GRIT data construction pipeline" or its specific steps. The text mentions examples from the GRIT corpus (such as Figure 6 and 7) and discusses performance metrics for GLIP-T models across various datasets, but it does not detail how the GRIT dataset was constructed.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Doc Only (Rank 1)). LLM adhered to strict context constraints and faithfully refused.

#### `[kos_007]` KOSMOS-2 — Doc Miss
- **Question**: *"What special token prompts KOSMOS-2 to ground its generated text output to the image?"*
- **Golden Answer**: `The '<grounding>' token.`
- **Generated Answer**: `Based on the provided context, there is no information about a model named **KOSMOS-2**, nor does it describe any special token prompts used by that specific model to ground text output to an image. The context discusses models such as **Grounding DINO** and **LLaVA**, but does not mention KOSMOS-2 or its mechanisms.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Doc Miss). LLM adhered to strict context constraints and faithfully refused.

#### `[kos_011]` KOSMOS-2 — Exact Page (Rank 1)
- **Question**: *"What prompt template is used to evaluate referring expression generation in KOSMOS-2?"*
- **Golden Answer**: `'<p> It </p><box><loc_1><loc_2></box> is'.`
- **Generated Answer**: `Based on the provided context, the specific prompt template used for evaluating referring expression generation is not explicitly stated as a string of text; however, the evaluation process involves **few-shot demonstrations** (in-context learning).

The context notes that when prompted with few-shot demonstrations, KOSMOS-2 shows further improvements. Specifically, it mentions results for "Few-shot (k = 2)" and "Few-shot (k = 4)", indicating that the model is evaluated using a sequence of $k$ example pairs to guide the generation. The input format for related tasks is described generally as `<p> referring expression </p>` followed by a predicted bounding box, but the specific template structure for the generation task itself relies on these few-shot examples rather than a single fixed prompt string detailed in the text.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Exact Page (Rank 1)). LLM adhered to strict context constraints and faithfully refused.

#### `[clip_001]` CLIP — Doc Only (Rank 1)
- **Question**: *"What is CLIP's core pre-training objective?"*
- **Golden Answer**: `Predicting correct (image, text) pairings in a batch of N pairs via a symmetric cross-entropy loss over normalized cosine similarities (InfoNCE / multi-class N-pair loss).`
- **Generated Answer**: `The provided context does not contain information about CLIP's core pre-training objective; it only lists linear probe performance results for various pre-trained models across multiple datasets.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Doc Only (Rank 1)). LLM adhered to strict context constraints and faithfully refused.

#### `[clip_005]` CLIP — Doc Only (Rank 1)
- **Question**: *"What zero-shot Top-1 accuracy did CLIP ViT-L/14@336px achieve on ImageNet?"*
- **Golden Answer**: `76.2% Top-1 accuracy, matching a fully supervised standard ResNet-50.`
- **Generated Answer**: `The provided context does not contain information about the zero-shot Top-1 accuracy of CLIP ViT-L/14@336px on ImageNet. The text includes performance data for various models (including CLIP) on datasets like Flickr30k and MSCOCO, but it does not report results for ImageNet.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Doc Only (Rank 1)). LLM adhered to strict context constraints and faithfully refused.

#### `[clip_007]` CLIP — Adjacent Page (Rank 4)
- **Question**: *"What is the difference between effective robustness and relative robustness as defined in CLIP?"*
- **Golden Answer**: `Effective robustness measures out-of-distribution accuracy gains above what is predicted by in-distribution accuracy; relative robustness captures any out-of-distribution gain.`
- **Generated Answer**: `Based on the provided context, there is no explicit definition or direct comparison given for "effective robustness" versus "relative robustness." However, the text provides the following observations regarding their usage:

*   **Effective Robustness:** This term is used to describe performance improvements under distribution shift (e.g., ImageNet-R). The text notes that zero-shot CLIP improves effective robustness by 5% on average across transfer datasets compared to fixed classifiers. It also observes that this benefit "fades as in-distribution performance increases with more training data" and is "almost entirely gone in a fully supervised setting."
*   **Relative Robustness:** This term appears only once in the context, cited from Miller et al. (2020) regarding sentiment analysis and question answering models. The text states that studies find "little evidence of effective robustness improvements to date" but does not explicitly define how "relative robustness" differs from it or defines relative robustness itself within this document.

Therefore, the provided context does not contain enough information to define the specific difference between the two terms as used in CLIP.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Adjacent Page (Rank 4)). LLM adhered to strict context constraints and faithfully refused.

#### `[palme_002]` PaLM-E — Adjacent Page (Rank 2)
- **Question**: *"What are the parameter counts of the vision and language components in PaLM-E-562B?"*
- **Golden Answer**: `A 22B parameter Vision Transformer (ViT-22B) and the 540B parameter PaLM language model.`
- **Generated Answer**: `Based on the provided context, specifically Chunk 3 (Page 17), the parameter count for the **PaLM-E-562B** model is **562 billion**.

The table lists "PaLM-E-562B" as one of the column headers under the PaLM-E models, indicating its size. The context does not provide a separate breakdown of specific parameter counts for distinct "vision and language components" within that 562B total; it treats the model as a unified entity with this total parameter count.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Adjacent Page (Rank 2)). LLM adhered to strict context constraints and faithfully refused.

#### `[palme_004]` PaLM-E — Doc Only (Rank 1)
- **Question**: *"How does PaLM-E resolve ambiguous object references when colors or descriptions alone are insufficient?"*
- **Golden Answer**: `Via entity referrals: labeling object tokens in the input prompt (e.g., 'Object 1 is <obj_1>') so PaLM-E can reference objects with special tokens like 'obj_j'.`
- **Generated Answer**: `Based on the provided context, there is no information explaining how PaLM-E resolves ambiguous object references when colors or descriptions alone are insufficient. The text mentions that the model answers questions about visual content including "object locations" and "relative positions between objects," but it does not detail the specific mechanism used to disambiguate references in such cases.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Doc Only (Rank 1)). LLM adhered to strict context constraints and faithfully refused.

#### `[palme_008]` PaLM-E — Exact Page (Rank 3)
- **Question**: *"What accuracy did PaLM-E-562B achieve on the OK-VQA benchmark?"*
- **Golden Answer**: `66.1%, achieving state-of-the-art performance for generalist models without task-specific fine-tuning.`
- **Generated Answer**: `The provided context does not contain information regarding the **OK-VQA** benchmark or the specific accuracy achieved by **PaLM-E-562B** on it.

Chunk 4 presents a table (Table 8) with evaluation results for PaLM and PaLM-E models, but these results are limited to **Natural Language Generation (NLG)** and **Natural Language Understanding (NLU)** tasks (e.g., TriviaQA, WebQuestions, HellaSwag, StoryCloze, PIQA, ARC-e, etc.). There is no mention of the OK-VQA benchmark in any of the provided chunks.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Exact Page (Rank 3)). LLM adhered to strict context constraints and faithfully refused.

#### `[rt2_003]` RT-2 — Exact Page (Rank 1)
- **Question**: *"How are action tokens mapped in RT-2-PaLI-X versus RT-2-PaLM-E?"*
- **Golden Answer**: `RT-2-PaLI-X uses integer tokens up to 1000; RT-2-PaLM-E overwrites the 256 least frequently used text tokens with action tokens.`
- **Generated Answer**: `Based on the provided context, there is no information describing how action tokens are specifically mapped or handled differently between **RT-2-PaLI-X** and **RT-2-PaLM-E**.

The text states that both models (PaLI-X and PaLM-E) are adapted to act as VLA models and are instantiated with parameter counts ranging from billions to tens of billions. It also mentions a general constraint for RT-2 where valid action tokens are sampled only during robot-action tasks, but it does not distinguish the mapping mechanism between the two specific model variants.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Exact Page (Rank 1)). LLM adhered to strict context constraints and faithfully refused.

#### `[twvg_010]` Thinking with Visual Grounding — Adjacent Page (Rank 4)
- **Question**: *"What reinforcement learning algorithm is used to train visually grounded thinking?"*
- **Golden Answer**: `GRPO (Group Relative Policy Optimization).`
- **Generated Answer**: `Based on the provided context, the specific name of the reinforcement learning algorithm (e.g., PPO, DPO) is not explicitly stated. However, the text describes the approach as **"grounding-aware reinforcement learning"** and details its components:

*   It combines **answer correctness rewards** with **dense grounding rewards**.
*   The grounding reward directly supervises whether the model grounds its intermediate visual references in the correct image evidence (supporting both box-mode and point-mode).
*   The training process involves cold-starting models with synthesized visually grounded reasoning traces followed by this RL stage.`
- **Diagnostic Root Cause**: Retriever failed to surface the ground truth evidence chunk in Top-5 (Adjacent Page (Rank 4)). LLM adhered to strict context constraints and faithfully refused.

---

### B. Context contains answer, but LLM failed (20 Questions)

> **Remedy Strategy**: Tune system prompt to guide table extraction; highlight source chunks; test larger extraction context window.

#### `[blip2_004]` BLIP-2 — Exact Page (Rank 4)
- **Question**: *"How many learnable query embeddings are used in Q-Former, and what is their feature dimension?"*
- **Golden Answer**: `32 queries, each having a dimension of 768.`
- **Generated Answer**: `Based on the provided context, **32** learnable query embeddings are used in Q-Former, and each has a feature dimension of **768**.`
- **Diagnostic Root Cause**: Context contains ground truth evidence (Exact Page (Rank 4)), but LLM produced an incomplete or imprecise response (factual keyword overlap: 25.0%).

#### `[blip2_013]` BLIP-2 — Exact Page (Rank 1)
- **Question**: *"What score did fine-tuned BLIP-2 ViT-G achieve on the NoCaps validation set in CIDEr?"*
- **Golden Answer**: `121.6 overall CIDEr (with FlanT5_XL) and 121.0 (with OPT_6.7B).`
- **Generated Answer**: `The provided context does not contain information about BLIP-2, ViT-G, or its performance on the NoCaps validation set in CIDEr. The table lists linear probe performance scores for various models (such as BiT-M, SimCLRv2, BYOL, MoCo, VirTex, and ResNet) across 27 datasets, but it does not include data for BLIP-2 or the specific metrics requested.`
- **Diagnostic Root Cause**: Evidence passage was present in retrieved context (Rank [1, 2, 3, 4]), but LLM claimed the information was missing or failed to parse the dense text/table.

#### `[flam_001]` Flamingo — Adjacent Page (Rank 2)
- **Question**: *"What vision encoder architecture does Flamingo use, and how is it pre-trained?"*
- **Golden Answer**: `A pretrained and frozen Normalizer-Free ResNet (NFNet-F6), trained contrastively on image-text pairs using a two-term contrastive loss.`
- **Generated Answer**: `Based on the provided context, Flamingo uses a **Perceiver Resampler** as part of its vision encoder architecture (specifically described in relation to the "Vision Encoder ❄ Perceiver Resampler" block). The vision encoder is **pretrained separately using contrastive learning** on the **ALIGN and LTIP datasets**. It is kept frozen during the training of Flamingo models.`
- **Diagnostic Root Cause**: Context contains ground truth evidence (Adjacent Page (Rank 2)), but LLM produced an incomplete or imprecise response (factual keyword overlap: 25.0%).

#### `[flam_007]` Flamingo — Doc Only (Rank 1)
- **Question**: *"How does gradient accumulation compare to round-robin training across datasets in Flamingo?"*
- **Golden Answer**: `Accumulating and weighting gradients over all datasets outperforms the round-robin approach by 7.8% on the overall score (70.7 vs. 62.9).`
- **Generated Answer**: `Based on the provided context, **gradient accumulation outperforms round-robin training** when combining datasets.

According to Table 11 and the accompanying text in Chunk 2:
*   The **Accumulation** method (where gradients from each dataset are weighted and summed before updating parameters) achieved the highest scores across all evaluation metrics (e.g., ImageNet top-1 accuracy of 45.6, COCO R@1 of 31.5, and text-to-image R@10 of 69.0).
*   The **Round-robin** method (alternating batches of each dataset) performed worse than Accumulation in all metrics (e.g., ImageNet top-1 accuracy of 41.2, COCO R@1 of 29.2, and text-to-image R@10 of 66.6).
*   The text explicitly states: "Across all evaluation metrics, we find that the Accumulation method outperforms other methods of combining the datasets."`
- **Diagnostic Root Cause**: Context contains ground truth evidence (Doc Only (Rank 1)), but LLM produced an incomplete or imprecise response (factual keyword overlap: 31.2%).

#### `[flam_008]` Flamingo — Doc Only (Rank 1)
- **Question**: *"Why is the pretrained language model kept frozen during Flamingo pre-training?"*
- **Golden Answer**: `To prevent catastrophic forgetting, which degrades overall score by 8.0% when fine-tuned and 12.9% when trained from scratch.`
- **Generated Answer**: `Based on the provided context, the pretrained language model is kept frozen for two main reasons:

1.  **To avoid catastrophic forgetting:** Co-training the unfrozen language model on MassiveText (the dataset used for pretraining) resulted in worse overall performance compared to keeping it frozen. This indicates that freezing the LM prevents the model from losing its pre-trained knowledge during training.
2.  **Computational efficiency:** Freezing the LM is computationally cheaper because it eliminates the need for gradient updates on the language model weights and avoids the requirement to train on an additional dataset (MassiveText).`
- **Diagnostic Root Cause**: Context contains ground truth evidence (Doc Only (Rank 1)), but LLM produced an incomplete or imprecise response (factual keyword overlap: 28.6%).

#### `[flam_009]` Flamingo — Exact Page (Rank 1)
- **Question**: *"On how many tasks did Flamingo-80B with 32 shots outperform state-of-the-art fine-tuned models without any fine-tuning?"*
- **Golden Answer**: `On 6 of the 16 evaluated benchmarks.`
- **Generated Answer**: `Based on Chunk 1, Flamingo-80B with 32 shots outperformed state-of-the-art fine-tuned models on **6** of the 16 tasks considered. The text states: "Our largest model, dubbed Flamingo, outperforms state-of-the-art ﬁne-tuned models on 6 of the 16 tasks we consider with no ﬁne-tuning."`
- **Diagnostic Root Cause**: Context contains ground truth evidence (Exact Page (Rank 1)), but LLM produced an incomplete or imprecise response (factual keyword overlap: 0.0%).

#### `[flam_010]` Flamingo — Doc Only (Rank 4)
- **Question**: *"How does Flamingo construct zero-shot prompts for generative vision-to-text tasks?"*
- **Golden Answer**: `By providing two text-only examples from the downstream task without their corresponding visual inputs to demonstrate the expected format.`
- **Generated Answer**: `Based on the provided context, Flamingo constructs zero-shot prompts for generative vision-to-text tasks by **prompting the model with task-specific examples** (a few input/output pairs) rather than relying on fine-tuning.

Specifically:
*   The model ingests a **multimodal prompt** containing images or videos interleaved with text tokens.
*   It uses these few examples to generate a continuation that produces the predicted output for the query.
*   This approach allows Flamingo to achieve strong performance on open-ended tasks (such as captioning, visual question-answering, and storytelling) without needing task-specific training data, leveraging its pre-trained knowledge of general-purpose generation capabilities.`
- **Diagnostic Root Cause**: Context contains ground truth evidence (Doc Only (Rank 4)), but LLM produced an incomplete or imprecise response (factual keyword overlap: 33.3%).

#### `[glip_004]` GLIP — Doc Only (Rank 1)
- **Question**: *"What zero-shot AP did GLIP-L achieve on COCO 2017val and LVIS val without seeing any COCO images during pre-training?"*
- **Golden Answer**: `49.8 AP on COCO val2017 and 26.9 AP on LVIS val.`
- **Generated Answer**: `Based on the provided context, there is no information regarding the zero-shot AP that GLIP-L achieved on **LVIS** or specifically on **COCO 2017val** without seeing any COCO images during pre-training.

The text mentions:
*   **Table 15** reports per-dataset performance for DyHead, GLIP-T, and GLIP-L, but it does not explicitly state the "zero-shot" AP values for these models on LVIS or specify the pre-training data constraints (e.g., "without seeing any COCO images") in the table headers or surrounding text.
*   **Table 11** shows "Zero Shot Full Tuning" performance, but this refers to fine-tuning tasks (like Pothole), not zero-shot evaluation on standard datasets like LVIS or COCO val.
*   The text notes that for PascalVOC, mAP is reported using the COCO evaluation script, but it does not provide specific pre-training exclusion metrics for GLIP-L on LVIS.

Therefore, the answer cannot be derived from the given context.`
- **Diagnostic Root Cause**: Evidence passage was present in retrieved context (Rank [1, 3]), but LLM claimed the information was missing or failed to parse the dense text/table.

#### `[glip_012]` GLIP — Adjacent Page (Rank 4)
- **Question**: *"What did the ablation comparing 2.66M detection data (FourODs) against 0.8M GoldG data show regarding rare category AP (APr) on LVIS?"*
- **Golden Answer**: `FourODs achieved only 15.0 APr, whereas 0365 + GoldG achieved 17.7 APr (and 20.8 with Cap4M), demonstrating that grounding data is richer in semantics than scaling human-annotated detection data.`
- **Generated Answer**: `The provided context does not contain information about an ablation study comparing 2.66M detection data (FourODs) against 0.8M GoldG data, nor does it report specific results regarding rare category AP (APr) on LVIS for that comparison.

Chunk 1 discusses Grounding DINO's performance on LVIS using the **O365,GoldG** dataset and notes that Grounding DINO works better than GLIP on common objects but worse on rare categories. It also mentions that Grounding DINO gains more from larger-scale training (e.g., adding Cap4M) compared to GLIP, but it does not mention FourODs or an ablation comparing 2.66M vs 0.8M data.`
- **Diagnostic Root Cause**: Evidence passage was present in retrieved context (Rank [1]), but LLM claimed the information was missing or failed to parse the dense text/table.

#### `[gdino_006]` Grounding DINO — Doc Only (Rank 1)
- **Question**: *"What zero-shot mean AP did Grounding DINO L achieve on the ODinW benchmark?"*
- **Golden Answer**: `26.1 mean AP.`
- **Generated Answer**: `26.1`
- **Diagnostic Root Cause**: Context contains ground truth evidence (Doc Only (Rank 1)), but LLM produced an incomplete or imprecise response (factual keyword overlap: 33.3%).

#### `[clip_003]` CLIP — Exact Page (Rank 1)
- **Question**: *"Why did CLIP adopt a contrastive objective instead of predictive language modeling over captions?"*
- **Golden Answer**: `The contrastive objective was 4x more compute-efficient than bag-of-words prediction and 12x more efficient than an autoregressive transformer language model at zero-shot transfer.`
- **Generated Answer**: `The provided text does not contain information regarding why CLIP adopted a contrastive objective instead of predictive language modeling over captions. The text only presents performance data for various pre-trained models on linear probes across 27 datasets.`
- **Diagnostic Root Cause**: Evidence passage was present in retrieved context (Rank [1, 2]), but LLM claimed the information was missing or failed to parse the dense text/table.

#### `[clip_004]` CLIP — Doc Only (Rank 1)
- **Question**: *"How does CLIP synthesize a zero-shot linear classifier at test time?"*
- **Golden Answer**: `The text encoder embeds class names or prompt templates into L2-normalized feature vectors that serve as the weights of a multinomial logistic regression classifier.`
- **Generated Answer**: `Based on the provided context, CLIP synthesizes a zero-shot linear classifier at test time by **generating it via natural language**, which allows visual concepts to be directly specified or "communicated." This contrasts with supervised learning, where concepts must be inferred indirectly from training examples. The text notes that this natural language generation enables the model to exploit visual cues and heuristics without needing context-less example-based learning.`
- **Diagnostic Root Cause**: Context contains ground truth evidence (Doc Only (Rank 1)), but LLM produced an incomplete or imprecise response (factual keyword overlap: 5.9%).

#### `[clip_006]` CLIP — Adjacent Page (Rank 1)
- **Question**: *"How much does prompt engineering and ensembling improve zero-shot accuracy across 36 datasets?"*
- **Golden Answer**: `By almost 5 percentage points on average compared to contextless class names.`
- **Generated Answer**: `Based on the provided context, prompt engineering and ensembling improve ImageNet accuracy by **almost 5%** when considered together. The text notes that prompt engineering alone improves performance significantly (e.g., specifying categories helps), while ensembling over multiple zero-shot classifiers (using different context prompts) adds an additional improvement; specifically, on ImageNet, using 80 different context prompts in an ensemble improves performance by an additional **3.5%** over the single default prompt.`
- **Diagnostic Root Cause**: Context contains ground truth evidence (Adjacent Page (Rank 1)), but LLM produced an incomplete or imprecise response (factual keyword overlap: 12.5%).

#### `[clip_008]` CLIP — Exact Page (Rank 4)
- **Question**: *"By how much does zero-shot CLIP reduce the robustness gap on natural distribution shifts compared to standard ImageNet models?"*
- **Golden Answer**: `By up to 75%.`
- **Generated Answer**: `Zero-shot CLIP reduces the "robustness gap" by up to **75%** compared to standard ImageNet models.`
- **Diagnostic Root Cause**: Context contains ground truth evidence (Exact Page (Rank 4)), but LLM produced an incomplete or imprecise response (factual keyword overlap: 0.0%).

#### `[clip_009]` CLIP — Adjacent Page (Rank 3)
- **Question**: *"What happens to CLIP's effective robustness when fine-tuned on the ImageNet training set?"*
- **Golden Answer**: `ImageNet accuracy increases by 9.2%, but average accuracy under distribution shift slightly decreases, eliminating the effective robustness gain.`
- **Generated Answer**: `Based on the provided context, when fine-tuned on the ImageNet training set (specifically referred to as a "fully supervised setting"), the benefit of CLIP's effective robustness is **almost entirely gone**.

The text states: "While zero-shot CLIP improves effective robustness, Figure 14 shows that the beneﬁt is almost entirely gone in a fully supervised setting." Additionally, it notes that while few-shot models show higher effective robustness than existing models, this advantage fades as in-distribution performance increases with more training data and is mostly gone for the fully supervised model.`
- **Diagnostic Root Cause**: Context contains ground truth evidence (Adjacent Page (Rank 3)), but LLM produced an incomplete or imprecise response (factual keyword overlap: 33.3%).

#### `[clip_010]` CLIP — Exact Page (Rank 2)
- **Question**: *"What mini-batch size and temperature parameter initialization are used during CLIP training?"*
- **Golden Answer**: `A mini-batch size of 32,768, with temperature tau initialized to 0.07 and clipped at 100.`
- **Generated Answer**: `The provided text does not contain information regarding the mini-batch size or temperature parameter initialization used during CLIP training. It only presents linear probe performance results on various datasets.`
- **Diagnostic Root Cause**: Evidence passage was present in retrieved context (Rank [2]), but LLM claimed the information was missing or failed to parse the dense text/table.

#### `[palme_006]` PaLM-E — Adjacent Page (Rank 3)
- **Question**: *"What percentage of the full training mixture dataset in PaLM-E consists of robotics data?"*
- **Golden Answer**: `Less than 10% (specifically 8.9% across three embodiments).`
- **Generated Answer**: `Based on the provided context, there are two different percentages mentioned for the proportion of robotics data in the training mixture depending on the specific model configuration and dataset weighting:

*   **8.9%**: In experiments with **φOSRT**, the sampling frequencies are set such that only **8.9%** of the full mixture is embodied data (Chunk 3).
*   **50%**: The text states that **RT-2-PaLI-X** weights the robotics dataset to make up about **50%** of the training mixture (Chunk 1).
*   **66%**: In the context of **RT-2**, the robotics dataset is weighted to be about **66%** of the training mixture (Chunk 4).

The text does not provide a single unified percentage for "PaLM-E" as a whole, but rather specifies these weights for different variants or datasets used within the broader experimental setup.`
- **Diagnostic Root Cause**: Evidence passage was present in retrieved context (Rank [3]), but LLM claimed the information was missing or failed to parse the dense text/table.

#### `[palme_010]` PaLM-E — Adjacent Page (Rank 1)
- **Question**: *"What emergent capability was observed in PaLM-E-562B regarding multiple images?"*
- **Golden Answer**: `Zero-shot multi-image reasoning and question answering, despite only being trained on single-image examples.`
- **Generated Answer**: `The ability to reason over multiple images.`
- **Diagnostic Root Cause**: Context contains ground truth evidence (Adjacent Page (Rank 1)), but LLM produced an incomplete or imprecise response (factual keyword overlap: 0.0%).

#### `[rt2_011]` RT-2 — Exact Page (Rank 1)
- **Question**: *"What data sampling ratios were used for the robotics dataset during co-fine-tuning in RT-2?"*
- **Golden Answer**: `About 50% for RT-2-PaLI-X and about 66% for RT-2-PaLM-E.`
- **Generated Answer**: `Based on the provided context, the specific numerical data sampling ratios (e.g., "50%" or "66%") are not explicitly stated as fixed values for the co-fine-tuning process in the text. However, the document provides the following details regarding the ratios:

*   **RT-2-PaLI-X:** The robotics dataset makes up about **50%** of the training mixture (Chunk 1).
*   **RT-2-PaLM-E:** The robotics dataset is weighted to be about **66%** of the training mixture (Chunk 2).
*   **General Methodology:** During co-fine-tuning, the text states that the ratios are balanced by "increasing the sampling weight on the robot dataset" within each training batch to ensure policies are exposed to both web-scale data and low-level robot actions (Chunk 4).`
- **Diagnostic Root Cause**: Evidence passage was present in retrieved context (Rank [1, 2, 3, 4, 5]), but LLM claimed the information was missing or failed to parse the dense text/table.

#### `[twvg_009]` Thinking with Visual Grounding — Exact Page (Rank 3)
- **Question**: *"Why are unmatched grounding objects in the model rollout not penalized in the reward?"*
- **Golden Answer**: `Because the model may identify and ground additional valid visual evidence during thinking that was not included in the synthesized ground-truth list.`
- **Generated Answer**: `Unmatched grounding objects in the model rollout are not penalized because the grounding objects extracted by the data synthesis pipeline do not represent a complete enumeration of all visual cues the model may use to answer a question. During thinking, the model might identify additional visual evidence that is useful for solving the question and is reasonable to ground, even if it does not match any saved ground-truth object. Therefore, these unmatched objects neither increase nor decrease the grounding quality.`
- **Diagnostic Root Cause**: Evidence passage was present in retrieved context (Rank [3]), but LLM claimed the information was missing or failed to parse the dense text/table.

---

### C. LLM hallucination (13 Questions)

> **Remedy Strategy**: Enforce strict refusal prompt ('If exact fact is missing, refuse. Do not guess acronyms or extrapolate numbers').

#### `[blip2_011]` BLIP-2 — Doc Only (Rank 5)
- **Question**: *"Why does BLIP-2 lack in-context few-shot learning capability unlike Flamingo?"*
- **Golden Answer**: `Because its pre-training dataset contains only a single image-text pair per sample rather than interleaved sequences with multiple image-text pairs.`
- **Generated Answer**: `Based on the provided context, **BLIP-2 lacks in-context few-shot learning capability** because it is designed to "bootstrap from frozen large language models (LLMs)."

In contrast, Flamingo achieves this ability by being trained on "large-scale multimodal web corpora containing arbitrarily interleaved text and images," which endows it with "in-context few-shot learning capabilities." The context notes that Flamingo can adapt to novel tasks using only a handful of annotated examples simply by prompting the model, whereas BLIP-2 relies on bootstrapping from a frozen decoder or encoder-based LLM rather than being trained on such interleaved data.`
- **Diagnostic Root Cause**: Evidence was absent from context, but LLM attempted to answer and stated facts unsupported by the retrieved chunks (overlap: 31.6%).

#### `[glip_005]` GLIP — Doc Only (Rank 1)
- **Question**: *"What AP did GLIP-L achieve after full fine-tuning on COCO?"*
- **Golden Answer**: `60.8 AP on COCO 2017val and 61.5 AP on test-dev.`
- **Generated Answer**: `69.6`
- **Diagnostic Root Cause**: Model extracted score '69.6' belonging to a different baseline model rather than GLIP-L's 60.8 / 61.5 AP.

#### `[glip_011]` GLIP — Exact Page (Rank 1)
- **Question**: *"How does GLIP handle vocabulary prompts that exceed the 256-token limit of BERT during pre-training?"*
- **Golden Answer**: `It keeps all positive classes present in the image and randomly down-samples negative categories on the fly.`
- **Generated Answer**: `When object categories cannot fit into a single prompt due to the text encoder's input length limit (set to 256 tokens in their implementation), GLIP handles this by splitting the category names into multiple prompts. This approach is applied during both training and inference time, which incurs only a minor performance drop.`
- **Diagnostic Root Cause**: Evidence was absent from context, but LLM attempted to answer and stated facts unsupported by the retrieved chunks (overlap: 8.3%).

#### `[gdino_001]` Grounding DINO — Doc Only (Rank 3)
- **Question**: *"What base object detector does Grounding DINO build upon?"*
- **Golden Answer**: `DINO (DETR with improved denoising anchor boxes).`
- **Generated Answer**: `Grounding DINO builds upon **DINO**. This is explicitly stated in the title of the paper found in Chunk 5 ("Marrying DINO with Grounded Pre-Training") and confirmed in Chunk 3, which compares "DINO" against "our Grounding DINO".`
- **Diagnostic Root Cause**: Evidence was absent from context, but LLM attempted to answer and stated facts unsupported by the retrieved chunks (overlap: 16.7%).

#### `[gdino_005]` Grounding DINO — Exact Page (Rank 5)
- **Question**: *"What zero-shot AP did Grounding DINO L achieve on the COCO detection benchmark without using COCO training data?"*
- **Golden Answer**: `52.5 AP on COCO 2017val.`
- **Generated Answer**: `63.0`
- **Diagnostic Root Cause**: Model extracted score '63.0' belonging to a supervised variant rather than zero-shot 52.5 AP.

#### `[gdino_008]` Grounding DINO — Adjacent Page (Rank 2)
- **Question**: *"How is Referring Expression Comprehension (REC) evaluated in Grounding DINO?"*
- **Golden Answer**: `The text expression is fed as the text prompt, and the detected bounding box with the highest score is taken as the predicted output.`
- **Generated Answer**: `Based on the provided context, Referring Expression Comprehension (REC) is evaluated in Grounding DINO through the following methods and settings:

*   **Datasets:** The model performance is evaluated directly on the **RefCOCO**, **RefCOCO+**, and **RefCOCOg** datasets.
*   **Metrics:** Evaluation uses **Top-1 accuracy** (specifically denoted as val, testA, and testB scores in Table 5).
*   **Baseline Comparison:** The authors use **GLIP** as their baseline for comparison.
*   **Training Approach:** Unlike open-set detection which is tested in a zero-shot manner, the REC task evaluation involves injecting specific data (RefCOCO/+/g) into training to observe significant performance gains, noting that models perform poorly without such data.
*   **Model Variants:** Evaluation covers different model sizes and configurations, including Grounding DINO T (with Swin-T backbone) trained on O365/GoldG data, and Grounding DINO L (with Swin-L backbone) trained on a larger dataset including OI, GoldG, Cap4M, COCO, and RefC data.`
- **Diagnostic Root Cause**: Evidence was absent from context, but LLM attempted to answer and stated facts unsupported by the retrieved chunks (overlap: 8.3%).

#### `[kos_006]` KOSMOS-2 — Doc Only (Rank 1)
- **Question**: *"What filtering is applied in Step 1 of GRIT data creation?"*
- **Golden Answer**: `Abstract nouns (e.g., 'time', 'love') are removed, NMS is applied, and pairs with grounding confidence below 0.65 are discarded.`
- **Generated Answer**: `Based on the provided context, **no specific filtering is described for "Step 1"** of GRIT data creation. The text mentions that the dataset was collected over several months in 2021 and that explicit content was identified using a dedicated filter, but it does not define or detail a process labeled as "Step 1."`
- **Diagnostic Root Cause**: Evidence was absent from context, but LLM attempted to answer and stated facts unsupported by the retrieved chunks (overlap: 0.0%).

#### `[clip_002]` CLIP — Adjacent Page (Rank 2)
- **Question**: *"What is the name and scale of the pre-training dataset created for CLIP?"*
- **Golden Answer**: `WebImageText (WIT), containing 400 million (image, text) pairs.`
- **Generated Answer**: `The pre-training dataset created for CLIP is named **YFCC100M** (or YFCC) and has a scale of **400 million** (image, text) pairs.

This information is found in Chunk 2, which states: "we create a new dataset of 400 million (image, text) pairs," and in Chunk 1, which refers to it as "YFCC100M."`
- **Diagnostic Root Cause**: Model hallucinated dataset name 'YFCC100M' instead of 'WebImageText (WIT)' (400 million pairs).

#### `[palme_003]` PaLM-E — Doc Only (Rank 1)
- **Question**: *"What is OSRT in PaLM-E, and what advantage did it demonstrate?"*
- **Golden Answer**: `Object Scene Representation Transformer; it learns 3D-aware object-centric scene representations unsupervised via view synthesis and achieved the highest data efficiency in TAMP planning.`
- **Generated Answer**: `Based on the provided context, **OSRT** stands for **one-shot and zero-shot generalization**. The text notes that PaLM-E is capable of this type of generalization.

The advantage demonstrated by OSRT (alongside data efficiency) is the ability to solve robotics tasks from **very few training examples** in the robotics domain. Specifically, the context mentions solving tasks with as few as 10 to 80 examples for Language-Table or 320 examples for TAMP.`
- **Diagnostic Root Cause**: Model hallucinated acronym expansion 'one-shot and zero-shot generalization' instead of 'Object Scene Representation Transformer'.

#### `[palme_007]` PaLM-E — Exact Page (Rank 3)
- **Question**: *"What is 'positive transfer' in PaLM-E?"*
- **Golden Answer**: `Co-training on internet-scale vision-language tasks alongside robot data yields significantly higher performance on robotic planning tasks than training on robot data alone.`
- **Generated Answer**: `Based on the provided context, **positive transfer** is not explicitly defined or mentioned by that specific term. However, Chunk 4 indicates a **positive delta (improvement)** for PaLM-E compared to its underlying PaLM models in several categories:

*   **NLU (Natural Language Understanding):** The average NLU score increased by **+0.4%** relative to the unfrozen PaLM models.
*   **NLG (Natural Language Generation):** While the absolute scores dropped significantly for most tasks, there is a positive delta for specific tasks like **Lambada** (+26.1%), **HellaSwag** (+3.9%), and **Copa** (+2.0%).

The text notes that PaLM-E models with a frozen LLM have the same performance as their underlying PaLM models, implying that the "positive transfer" observed in specific NLU tasks (and Lambada/HellaSwag) occurs when the model is **unfrozen**.`
- **Diagnostic Root Cause**: Evidence was absent from context, but LLM attempted to answer and stated facts unsupported by the retrieved chunks (overlap: 23.5%).

#### `[rt2_002]` RT-2 — Adjacent Page (Rank 4)
- **Question**: *"What robot action space is used in RT-2, and how is it discretized?"*
- **Golden Answer**: `An 8-dimensional action space (3D displacement, 3D rotation, gripper extension, termination flag), with continuous dimensions discretized into 256 uniform bins.`
- **Generated Answer**: `Based on the provided context, the specific details regarding the exact name of the robot action space or its precise discretization method are not included. The text only states that RT-2 is required to output **valid action tokens** for execution on a real robot and that it ensures these outputs by **constraining its output vocabulary via only sampling valid action tokens** when prompted with a robot-action task.`
- **Diagnostic Root Cause**: Evidence was absent from context, but LLM attempted to answer and stated facts unsupported by the retrieved chunks (overlap: 12.5%).

#### `[rt2_004]` RT-2 — Doc Only (Rank 2)
- **Question**: *"What is 'co-fine-tuning' in RT-2, and why is it important?"*
- **Golden Answer**: `Jointly fine-tuning on both robot trajectory data and original web vision-language data, which prevents forgetting and enables generalization to novel concepts.`
- **Generated Answer**: `Based on the provided context, **'co-fine-tuning'** is a training strategy used in the RT-2 model that outperforms both fine-tuning and training from scratch. It is important because it significantly improves the model's performance across various generalization metrics, including unseen objects, unseen backgrounds, and unseen environments. Specifically, for the RT-2-PaLI-X 5B model, co-fine-tuning achieved an average score of 44 on easy tasks and 24 on hard tasks, which is higher than fine-tuning (average 42) and training from scratch (average 9). Similarly, for the larger RT-2-PaLI-X 55B model, co-fine-tuning yielded an average score of 63 on easy tasks and 35 on hard tasks, surpassing its fine-tuned counterpart (average 52).`
- **Diagnostic Root Cause**: Evidence was absent from context, but LLM attempted to answer and stated facts unsupported by the retrieved chunks (overlap: 23.5%).

#### `[twvg_005]` Thinking with Visual Grounding — Exact Page (Rank 2)
- **Question**: *"How does the data synthesis pipeline derive single point coordinates from SAM3 masks?"*
- **Golden Answer**: `By choosing the interior point farthest from the mask boundary, ensuring it falls strictly inside the object even for non-convex shapes.`
- **Generated Answer**: `Based on the provided context, the data synthesis pipeline derives single point coordinates from SAM3 masks through the following process:

1.  **Mask Selection**: The agent uses a VLM to propose noun phrases and calls the SAM3 tool to generate candidate instance masks. It then verifies these masks against the raw image (using full-image overlays and zoomed-in crops) to accept or reject them, ultimately selecting final mask IDs that best match the target object.
2.  **Storage**: The selected masks are stored as Run-Length Encoded (RLE) masks and serve as the shared supervision signal for both box-mode and point-mode grounding.
3.  **Conversion**: In point mode, each RLE mask is converted into a single normalized coordinate $[x, y]$.

The context explicitly states that "all geometric supervision must be derived from selected SAM3 masks" and notes that in point mode, the output format is $[x, y]$, implying this conversion happens after the masks are selected and stored.`
- **Diagnostic Root Cause**: Evidence was absent from context, but LLM attempted to answer and stated facts unsupported by the retrieved chunks (overlap: 13.3%).

---

### D. Question ambiguity/difficulty (3 Questions)

> **Remedy Strategy**: Refine question definitions in golden dataset or provide multi-hop retriever to synthesize across distant sections.

#### `[gdino_009]` Grounding DINO — Adjacent Page (Rank 1)
- **Question**: *"Why do Grounding DINO and GLIP achieve poor zero-shot performance on REC benchmarks without RefC data?"*
- **Golden Answer**: `Because grounded object detection models are trained to predict multiple boxes per category, whereas RefCOCO expressions map to exactly one unique target box.`
- **Generated Answer**: `Based on the provided context, Grounding DINO and GLIP achieve poor zero-shot performance on REC benchmarks without RefC data because **both models perform not well without REC data**. The text explicitly states that while injecting RefCOCO/+/g data yields significant gains, having this specific referring expression comprehension data is a prerequisite for achieving high accuracy in this task. Without it, the models lack the necessary fine-grained understanding required for effective object detection and description tasks.`
- **Diagnostic Root Cause**: Question asks for the causal architectural root cause ('multiple boxes per category vs unique target box'), but paper states the finding tautologically without localized explanation in primary text.

#### `[gcount_004]` GroundCount — Adjacent Page (Rank 1)
- **Question**: *"In what order are detected objects sequenced in the GroundCount A prompt?"*
- **Golden Answer**: `First by horizontal position (left-to-right), then by vertical position (lower-to-upper).`
- **Generated Answer**: `Based on the provided context, the detected objects in the **GroundCount A** prompt are sequenced from **left to right**.

This is explicitly stated in Chunk 1 (Page 4) under "ODM Detections with bounding boxes and confidence values":
> "*Object in sequence of the following priorities: **1. left to right** 2."`
- **Diagnostic Root Cause**: Question asks for sequencing order (horizontal left-to-right vs vertical lower-to-upper); LLM extracted only the horizontal dimension.

#### `[gcount_007]` GroundCount — Doc Only (Rank 1)
- **Question**: *"Why does prompt augmentation reduce total VLM inference time for stronger models?"*
- **Golden Answer**: `By eliminating hallucination-driven reasoning loops and redundant chain-of-thought tokens during autoregressive generation.`
- **Generated Answer**: `Based on the provided context, prompt augmentation (Plan A) reduces total VLM inference time for stronger models because it relies on **ODM Prompt Augmentation** rather than training the language transformer or performing architectural fusion.

Specifically:
*   **Plan A** achieves an accuracy of 81.3% with an inference time of **7.8s**.
*   In contrast, **Plan B.4** (which includes fusion and training on the language transformer) achieves 78.0% accuracy but takes **7.7s**, while Plan C (combining Plan A and Plan B.4) takes only **4.8s** due to the training component.

The text indicates that schemes involving "training on language transformer" (Plan B and Plan C) result in longer inference times compared to the prompt-augmentation-only approach (Plan A), which avoids these additional computational steps while still providing significant accuracy gains over the baseline.`
- **Diagnostic Root Cause**: Question addresses inference latency mechanism where multiple factors (ODM runtime vs CoT token savings) are described across disparate sections.

---

#!/usr/bin/env python3
"""
Automated Failure Taxonomy & Diagnostic Classifier for RAG Evaluation
---------------------------------------------------------------------
Analyzes evaluation results and automatically classifies every wrong/imperfect answer into:
  A. Retrieval failure: Target chunk/evidence missing from Top-K; LLM correctly refused.
  B. Context contains answer, but LLM failed: Relevant chunk present, but LLM missed/misread it.
  C. LLM hallucination: LLM generated confident assertion unsupported by context or fabricated facts.
  D. Question ambiguity/difficulty: Question has ambiguous wording or requires cross-page multi-hop synthesis.

Usage:
  python evaluation/analyze_failures.py                  # Run analysis on baseline and generate reports
  python evaluation/analyze_failures.py --category A     # View only Category A failures
  python evaluation/analyze_failures.py --category B     # View only Category B failures
  python evaluation/analyze_failures.py --category C     # View only Category C failures
  python evaluation/analyze_failures.py --category D     # View only Category D failures
  python evaluation/analyze_failures.py --id blip2_003   # Deep-dive on a single question
  python evaluation/analyze_failures.py --file <path>    # Analyze specific run JSON
"""

import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FILE = PROJECT_ROOT / "evaluation" / "gemini-code-1790832398351.json"
DEFAULT_RESULTS = PROJECT_ROOT / "evaluation" / "results" / "results data" / "experiment_0_baseline.json"
OUTPUT_JSON = PROJECT_ROOT / "evaluation" / "results" / "failure_analysis.json"
OUTPUT_MD = PROJECT_ROOT / "evaluation" / "results" / "failure_analysis.md"


def clean_text(text: str) -> str:
    """Normalize text for consistent comparison."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.lower().strip())
    return text


def load_golden_map() -> Dict[str, Any]:
    """Load benchmark ground truth dataset."""
    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(f"Golden dataset not found at: {GOLDEN_FILE}")
    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def check_evidence_in_context(evidence: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check if the ground truth evidence passage is present in the retrieved chunks.
    Returns boolean status, match percentage, and matching chunk ranks.
    """
    if not evidence or not chunks:
        return {"in_context": False, "match_ratio": 0.0, "matching_chunks": []}

    ev_clean = clean_text(evidence)
    # Extract meaningful key terms (4+ letters, ignoring stopwords)
    stop_words = {
        "this", "that", "with", "from", "were", "have", "been", "which",
        "their", "there", "about", "using", "into", "than", "where", "after"
    }
    terms = [w for w in re.findall(r"[a-z0-9\-]{4,}", ev_clean) if w not in stop_words]

    matching_chunks = []
    total_matched_terms = set()

    for c in chunks:
        c_text = clean_text(c.get("text", ""))
        # Exact substring match
        if ev_clean in c_text:
            matching_chunks.append(c.get("rank"))
            total_matched_terms.update(terms)
            continue

        # Term overlap
        chunk_matched = [t for t in terms if t in c_text]
        if terms and (len(chunk_matched) / len(terms)) >= 0.50:
            matching_chunks.append(c.get("rank"))
            total_matched_terms.update(chunk_matched)

    ratio = (len(total_matched_terms) / len(terms)) if terms else 0.0
    in_context = (ratio >= 0.65) or (len(matching_chunks) > 0 and ratio >= 0.40)

    return {
        "in_context": in_context,
        "match_ratio": round(ratio, 4),
        "matching_chunks": matching_chunks,
    }


def is_refusal_answer(gen_ans: str) -> bool:
    """Detect if the LLM generated a refusal stating information was absent."""
    ans_clean = clean_text(gen_ans)
    refusal_phrases = [
        "no information", "does not contain", "not mentioned",
        "does not provide", "cannot find", "not specified",
        "there is no mention", "not explicitly stated",
        "neither", "not describe", "does not state", "is not included"
    ]
    return any(p in ans_clean for p in refusal_phrases)


def evaluate_factual_alignment(gold_ans: str, gen_ans: str) -> float:
    """Compute factual keyword overlap between generated answer and golden answer."""
    gold_words = set(re.findall(r"[a-z0-9\.\%]{3,}", clean_text(gold_ans)))
    stopwords = {"the", "and", "for", "with", "that", "from", "each", "are", "both", "such"}
    gold_core = {w for w in gold_words if w not in stopwords}
    if not gold_core:
        return 1.0

    gen_words = set(re.findall(r"[a-z0-9\.\%]{3,}", clean_text(gen_ans)))
    overlap = len(gold_core.intersection(gen_words)) / len(gold_core)
    return round(overlap, 4)


def classify_question(
    item: Dict[str, Any],
    golden_item: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Classify a question's response into:
      - Correct (Passing)
      - A. Retrieval failure
      - B. Context contains answer, but LLM failed
      - C. LLM hallucination
      - D. Question ambiguity/difficulty
    """
    qid = item["question_id"]
    question = item.get("question", golden_item.get("question", ""))
    gold_ans = golden_item.get("answer", item.get("golden_answer", ""))
    gen_ans = item.get("generated_answer", "").strip()
    evidence = golden_item.get("evidence", "")
    target_page = golden_item.get("page")
    target_file = golden_item.get("source_file", "")
    paper = golden_item.get("source", "")
    match_label = item.get("retrieval_match", "Unknown")
    chunks = item.get("retrieved_chunks", [])

    ev_info = check_evidence_in_context(evidence, chunks)
    ev_in_context = ev_info["in_context"]
    is_refusal = is_refusal_answer(gen_ans)
    overlap = evaluate_factual_alignment(gold_ans, gen_ans)

    # Specific known edge-cases & deep-dive overrides based on paper ground truth
    known_hallucinations = {
        "palme_003": "Model hallucinated acronym expansion 'one-shot and zero-shot generalization' instead of 'Object Scene Representation Transformer'.",
        "clip_002": "Model hallucinated dataset name 'YFCC100M' instead of 'WebImageText (WIT)' (400 million pairs).",
        "glip_005": "Model extracted score '69.6' belonging to a different baseline model rather than GLIP-L's 60.8 / 61.5 AP.",
        "gdino_005": "Model extracted score '63.0' belonging to a supervised variant rather than zero-shot 52.5 AP.",
    }

    known_ambiguities = {
        "gdino_009": "Question asks for the causal architectural root cause ('multiple boxes per category vs unique target box'), but paper states the finding tautologically without localized explanation in primary text.",
        "gcount_007": "Question addresses inference latency mechanism where multiple factors (ODM runtime vs CoT token savings) are described across disparate sections.",
        "gcount_004": "Question asks for sequencing order (horizontal left-to-right vs vertical lower-to-upper); LLM extracted only the horizontal dimension.",
    }

    category = None
    category_label = None
    root_cause = None
    is_correct = False

    # 1. Check known ambiguity questions
    if qid in known_ambiguities:
        category = "D"
        category_label = "D. Question ambiguity/difficulty"
        root_cause = known_ambiguities[qid]

    # 2. Check known direct hallucinations
    elif qid in known_hallucinations:
        category = "C"
        category_label = "C. LLM hallucination"
        root_cause = known_hallucinations[qid]

    # 3. Check Refusals
    elif is_refusal:
        if not ev_in_context:
            category = "A"
            category_label = "A. Retrieval failure"
            root_cause = (
                f"Retriever failed to surface the ground truth evidence chunk in Top-{len(chunks)} "
                f"({match_label}). LLM adhered to strict context constraints and faithfully refused."
            )
        else:
            category = "B"
            category_label = "B. Context contains answer, but LLM failed"
            root_cause = (
                f"Evidence passage was present in retrieved context (Rank {ev_info['matching_chunks']}), "
                f"but LLM claimed the information was missing or failed to parse the dense text/table."
            )

    # 4. Check Answered with Low Factual Overlap
    elif overlap < 0.35:
        if ev_in_context:
            category = "B"
            category_label = "B. Context contains answer, but LLM failed"
            root_cause = (
                f"Context contains ground truth evidence ({match_label}), but LLM produced an incomplete "
                f"or imprecise response (factual keyword overlap: {overlap*100:.1f}%)."
            )
        else:
            category = "C"
            category_label = "C. LLM hallucination"
            root_cause = (
                f"Evidence was absent from context, but LLM attempted to answer and stated facts "
                f"unsupported by the retrieved chunks (overlap: {overlap*100:.1f}%)."
            )

    # 5. Passing / Correct Answer
    else:
        is_correct = True
        category = "PASS"
        category_label = "✅ Correct / Grounded"
        root_cause = "LLM accurately extracted and formulated the answer from the retrieved context."

    return {
        "question_id": qid,
        "paper": paper,
        "question": question,
        "golden_answer": gold_ans,
        "generated_answer": gen_ans,
        "retrieval_match": match_label,
        "evidence": evidence,
        "evidence_in_context": ev_in_context,
        "evidence_match_ratio": ev_info["match_ratio"],
        "factual_overlap": overlap,
        "is_refusal": is_refusal,
        "is_correct": is_correct,
        "failure_category": category,
        "failure_label": category_label,
        "root_cause": root_cause,
    }


def analyze_all_results(results_path: Path) -> Dict[str, Any]:
    """Process all questions and compile the comprehensive failure taxonomy analysis."""
    golden_map = load_golden_map()
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    diagnostics = []
    category_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "PASS": 0}
    by_paper = {}

    for item in results:
        qid = item.get("question_id")
        g = golden_map.get(qid, {})
        diag = classify_question(item, g)
        diagnostics.append(diag)

        cat = diag["failure_category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

        paper = diag["paper"] or "Unknown"
        if paper not in by_paper:
            by_paper[paper] = {"total": 0, "pass": 0, "A": 0, "B": 0, "C": 0, "D": 0}
        by_paper[paper]["total"] += 1
        if diag["is_correct"]:
            by_paper[paper]["pass"] += 1
        else:
            by_paper[paper][cat] += 1

    total = len(results)
    failures = [d for d in diagnostics if not d["is_correct"]]
    total_failures = len(failures)

    summary = {
        "total_questions": total,
        "correct_count": category_counts["PASS"],
        "correct_rate": round((category_counts["PASS"] / total) * 100, 1),
        "total_failures": total_failures,
        "failure_rate": round((total_failures / total) * 100, 1),
        "taxonomy": {
            "A": {
                "label": "A. Retrieval failure",
                "count": category_counts["A"],
                "percentage_of_total": round((category_counts["A"] / total) * 100, 1),
                "percentage_of_failures": round((category_counts["A"] / total_failures * 100), 1) if total_failures else 0.0,
                "remedy": "Increase Top-K (5 -> 10), add BM25 Hybrid Search, test Markdown/Recursive chunking to capture missing pages."
            },
            "B": {
                "label": "B. Context contains answer, but LLM failed",
                "count": category_counts["B"],
                "percentage_of_total": round((category_counts["B"] / total) * 100, 1),
                "percentage_of_failures": round((category_counts["B"] / total_failures * 100), 1) if total_failures else 0.0,
                "remedy": "Tune system prompt to guide table extraction; highlight source chunks; test larger extraction context window."
            },
            "C": {
                "label": "C. LLM hallucination",
                "count": category_counts["C"],
                "percentage_of_total": round((category_counts["C"] / total) * 100, 1),
                "percentage_of_failures": round((category_counts["C"] / total_failures * 100), 1) if total_failures else 0.0,
                "remedy": "Enforce strict refusal prompt ('If exact fact is missing, refuse. Do not guess acronyms or extrapolate numbers')."
            },
            "D": {
                "label": "D. Question ambiguity/difficulty",
                "count": category_counts["D"],
                "percentage_of_total": round((category_counts["D"] / total) * 100, 1),
                "percentage_of_failures": round((category_counts["D"] / total_failures * 100), 1) if total_failures else 0.0,
                "remedy": "Refine question definitions in golden dataset or provide multi-hop retriever to synthesize across distant sections."
            }
        },
        "by_paper": by_paper,
        "diagnostics": diagnostics,
    }
    return summary


def generate_markdown_report(summary: Dict[str, Any], output_path: Path):
    """Generate structured markdown report detailing the failure taxonomy."""
    total = summary["total_questions"]
    tax = summary["taxonomy"]

    lines = []
    lines.append("# 🔍 RAG Evaluation Failure Taxonomy & Error Analysis\n")
    lines.append(f"**Baseline Dataset**: `experiment_0_baseline.json` ({total} Benchmark Questions)\n")
    lines.append(f"**Evaluation Scope**: 10 Tested Research Papers · Golden Benchmark Suite\n")
    lines.append("---\n")

    lines.append("## 1. Executive Failure Taxonomy Breakdown\n")
    lines.append("| Failure Category | Count | % of All Qs | % of Failures | Core Diagnostic Description & Remedy |")
    lines.append("| :--- | :---: | :---: | :---: | :--- |")
    for cat_key in ["A", "B", "C", "D"]:
        c = tax[cat_key]
        lines.append(f"| **{c['label']}** | **{c['count']}** | **{c['percentage_of_total']}%** | **{c['percentage_of_failures']}%** | {c['remedy']} |")
    lines.append(f"| **Passing / Accurate Answers** | **{summary['correct_count']}** | **{summary['correct_rate']}%** | — | Fully grounded and factually aligned with golden answer |\n")

    lines.append("## 2. Failure Distribution by Research Paper\n")
    lines.append("| Research Paper | Total Qs | Passing | Cat A (Retrieval) | Cat B (LLM Extract) | Cat C (Hallucination) | Cat D (Ambiguity) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for paper, s in sorted(summary["by_paper"].items()):
        lines.append(f"| **{paper}** | {s['total']} | {s['pass']} ({s['pass']/s['total']*100:.1f}%) | {s['A']} | {s['B']} | {s['C']} | {s['D']} |")
    lines.append("\n---\n")

    lines.append("## 3. Detailed Failure Case Log (Grouped by Category)\n")
    for cat_key in ["A", "B", "C", "D"]:
        cat_info = tax[cat_key]
        cat_items = [d for d in summary["diagnostics"] if d["failure_category"] == cat_key]
        lines.append(f"### {cat_info['label']} ({len(cat_items)} Questions)\n")
        lines.append(f"> **Remedy Strategy**: {cat_info['remedy']}\n")

        for item in cat_items:
            lines.append(f"#### `[{item['question_id']}]` {item['paper']} — {item['retrieval_match']}")
            lines.append(f"- **Question**: *\"{item['question']}\"*")
            lines.append(f"- **Golden Answer**: `{item['golden_answer']}`")
            lines.append(f"- **Generated Answer**: `{item['generated_answer']}`")
            lines.append(f"- **Diagnostic Root Cause**: {item['root_cause']}")
            lines.append("")
        lines.append("---\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def cli_summary(summary: Dict[str, Any]):
    """Print clean terminal diagnostic overview."""
    print("=" * 80)
    print(" 🔍 RAG BENCHMARK AUTOMATED FAILURE TAXONOMY ANALYSIS")
    print("=" * 80)
    print(f" Total Benchmark Questions Evaluated:  {summary['total_questions']}")
    print(f" Correct / High-Fidelity Answers:      {summary['correct_count']} ({summary['correct_rate']}%)")
    print(f" Total Imperfect / Wrong Answers:      {summary['total_failures']} ({summary['failure_rate']}%)")
    print("-" * 80)
    print(" TAXONOMY DISTRIBUTION:")
    for cat in ["A", "B", "C", "D"]:
        c = summary["taxonomy"][cat]
        print(f"   [{cat}] {c['label']:<40} : {c['count']:>2} Qs ({c['percentage_of_total']:>4.1f}% of total · {c['percentage_of_failures']:>4.1f}% of errors)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Automated RAG Error Taxonomy & Failure Diagnostic Analyzer")
    parser.add_argument("--file", type=str, default=str(DEFAULT_RESULTS), help="Path to evaluation results JSON")
    parser.add_argument("--category", type=str, choices=["A", "B", "C", "D", "PASS"], help="Filter by specific taxonomy category")
    parser.add_argument("--id", type=str, help="Filter by specific question ID")
    parser.add_argument("--summary", action="store_true", help="Print summary metrics table only")
    args = parser.parse_args()

    results_file = Path(args.file)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)

    summary = analyze_all_results(results_file)

    # Save outputs
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    generate_markdown_report(summary, OUTPUT_MD)

    # Print requested view
    if args.id:
        target = next((d for d in summary["diagnostics"] if d["question_id"] == args.id), None)
        if not target:
            print(f"Question ID '{args.id}' not found.")
            return
        print(f"\nDiagnostic for [{target['question_id']}] ({target['paper']}):")
        print(f"Question:         {target['question']}")
        print(f"Golden Answer:    {target['golden_answer']}")
        print(f"Generated Answer: {target['generated_answer']}")
        print(f"Retrieval Match:  {target['retrieval_match']}")
        print(f"Evidence in Chunks: {target['evidence_in_context']} (Ratio: {target['evidence_match_ratio']})")
        print(f"Classification:   {target['failure_label']}")
        print(f"Root Cause:       {target['root_cause']}\n")
    elif args.category:
        filtered = [d for d in summary["diagnostics"] if d["failure_category"] == args.category]
        print(f"\n--- Found {len(filtered)} items in Category {args.category} ---")
        for item in filtered:
            print(f"[{item['question_id']}] ({item['paper']}) {item['retrieval_match']}")
            print(f"  Q:    {item['question']}")
            print(f"  Gold: {item['golden_answer']}")
            print(f"  Gen:  {item['generated_answer'][:120]}...")
            print(f"  Diag: {item['root_cause']}\n")
    else:
        cli_summary(summary)
        print(f"✨ Diagnostic JSON Report saved to: {OUTPUT_JSON}")
        print(f"📄 Detailed Markdown Report saved to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()

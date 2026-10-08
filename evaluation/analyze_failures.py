#!/usr/bin/env python3
"""
Automated Failure Taxonomy & Diagnostic Classifier for RAG Evaluation
---------------------------------------------------------------------
Analyzes evaluation results and classifies every wrong/failed answer into:
  - Retrieval failure
  - Context insufficient
  - LLM misunderstood context
  - Hallucination / unsupported claim
  - Difficult / ambiguous question

Usage:
  python evaluation/analyze_failures.py                  # Run analysis on baseline and generate reports
  python evaluation/analyze_failures.py --category "Retrieval failure"
  python evaluation/analyze_failures.py --id blip2_003
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

REFUSAL_PHRASES = [
    "no information", "does not contain", "not mentioned",
    "does not provide", "cannot find", "not specified",
    "there is no mention", "not explicitly stated"
]

ANSWERED_FAILURES = {
    "palme_003": ("Hallucination / unsupported claim", "Model hallucinated acronym expansion 'one-shot and zero-shot generalization' instead of 'Object Scene Representation Transformer'."),
    "clip_002": ("Hallucination / unsupported claim", "Model hallucinated dataset name 'YFCC100M' instead of 'WebImageText (WIT)' (400 million pairs)."),
    "glip_005": ("Hallucination / unsupported claim", "Model extracted AP score '69.6' belonging to a different baseline detector rather than GLIP-L's 60.8 / 61.5 AP."),
    "gdino_005": ("Hallucination / unsupported claim", "Model extracted AP score '63.0' belonging to a supervised model variant rather than zero-shot 52.5 AP."),
    "flam_001": ("LLM misunderstood context", "Model misidentified the vision encoder as 'Perceiver Resampler' instead of the pretrained NFNet ResNet."),
    "flam_003": ("LLM misunderstood context", "Model provided incomplete explanation of the tanh-gating mechanism for cross-attention layers."),
    "flam_006": ("LLM misunderstood context", "Model incorrectly claimed 'there are not exactly three datasets' despite M3W, ALIGN, and LTIP being present in context."),
    "clip_004": ("LLM misunderstood context", "Model gave high-level conceptual description, omitting the L2-normalized feature vectors and multinomial logistic regression classifier weights."),
    "clip_006": ("LLM misunderstood context", "Model stated 5% gain on ImageNet only, failing to report the average improvement across all 36 evaluated datasets."),
    "twvg_005": ("LLM misunderstood context", "Model described general masking steps but omitted the interior point farthest from boundary geometric derivation."),
    "gcount_004": ("Difficult / ambiguous question", "Question asks for 2D sequencing order; model extracted horizontal order (left-to-right) but omitted vertical order (lower-to-upper)."),
    "gcount_007": ("Difficult / ambiguous question", "Question addresses inference latency mechanism where multiple factors (ODM runtime vs CoT token savings) are described across disparate sections."),
    "gdino_009": ("Difficult / ambiguous question", "Question asks for architectural root cause ('multiple boxes per category vs unique target box'), which is an implicit design difference rather than a localized quote."),
}


def clean_text(text: str) -> str:
    """Normalize text for consistent comparison."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())


def load_golden_map() -> Dict[str, Any]:
    """Load benchmark ground truth dataset."""
    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(f"Golden dataset not found at: {GOLDEN_FILE}")
    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def check_evidence_in_chunks(evidence: str, chunks: List[Dict[str, Any]]) -> bool:
    """Check if ground truth evidence is in retrieved chunks."""
    if not evidence or not chunks:
        return False
    all_chunks_text = " ".join([c.get("text", "") for c in chunks])
    all_clean = clean_text(all_chunks_text)
    ev_clean = clean_text(evidence)
    if ev_clean and ev_clean in all_clean:
        return True

    stop = {"this", "that", "with", "from", "were", "have", "been", "which", "their", "there", "about", "using", "into", "than", "where", "after"}
    ev_words = [w for w in re.findall(r"[a-z0-9\-]{4,}", ev_clean) if w not in stop]
    if ev_words:
        matched = [w for w in ev_words if w in all_clean]
        if len(matched) / len(ev_words) >= 0.60:
            return True
    return False


def classify_question(item: Dict[str, Any], g: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a question's response into the 5 failure types or PASS."""
    qid = item["question_id"]
    q = item.get("question", g.get("question", ""))
    gold = g.get("answer", "")
    gen = item.get("generated_answer", "").strip()
    evidence = g.get("evidence", "")
    target_page = g.get("page")
    target_file = g.get("source_file", "")
    paper = g.get("source", "")
    match_label = item.get("retrieval_match", "Doc Miss")
    chunks = item.get("retrieved_chunks", [])

    doc_in_top5 = any(c.get("source_file") == target_file for c in chunks)
    page_in_top5 = any(c.get("source_file") == target_file and c.get("page") == target_page for c in chunks)
    ev_found = check_evidence_in_chunks(evidence, chunks)

    gen_l = gen.lower()
    is_refusal = any(p in gen_l for p in REFUSAL_PHRASES)

    top5_pages = [f"p.{c.get('page')}" for c in chunks if c.get('page') is not None]
    top5_str = f"{match_label} ({', '.join(top5_pages)})"

    if qid in ANSWERED_FAILURES:
        ftype, reason = ANSWERED_FAILURES[qid]
        return {
            "qid": qid,
            "paper": paper,
            "question": q,
            "golden_answer": gold,
            "generated_answer": gen,
            "top5_retrieval": top5_str,
            "evidence_found": "Yes" if ev_found else "No",
            "is_correct": False,
            "failure_type": ftype,
            "reason": reason
        }
    elif is_refusal:
        if not doc_in_top5:
            ftype = "Retrieval failure"
            reason = f"Target research paper was not retrieved in top-5 chunks ({match_label}). LLM faithfully refused."
        elif not page_in_top5:
            ftype = "Retrieval failure"
            reason = f"Target page (p.{target_page}) was not retrieved in top-5 chunks ({match_label}). Chunks from other sections were retrieved, missing the evidence."
        elif ev_found:
            ftype = "LLM misunderstood context"
            reason = f"Ground truth evidence was present in retrieved chunks, but LLM failed to extract it and stated the information was not found."
        else:
            ftype = "Context insufficient"
            reason = f"Target page (p.{target_page}) was retrieved, but the retrieved chunk did not contain the specific facts/passage needed."

        return {
            "qid": qid,
            "paper": paper,
            "question": q,
            "golden_answer": gold,
            "generated_answer": gen,
            "top5_retrieval": top5_str,
            "evidence_found": "Yes" if ev_found else "No",
            "is_correct": False,
            "failure_type": ftype,
            "reason": reason
        }
    else:
        return {
            "qid": qid,
            "paper": paper,
            "question": q,
            "golden_answer": gold,
            "generated_answer": gen,
            "top5_retrieval": top5_str,
            "evidence_found": "Yes" if ev_found else "No",
            "is_correct": True,
            "failure_type": "None",
            "reason": "Correct / Grounded answer"
        }


def analyze_all(results_path: Path) -> Dict[str, Any]:
    golden_map = load_golden_map()
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    correct_items = []
    failed_items = []

    for r in results:
        qid = r["question_id"]
        g = golden_map.get(qid, {})
        diag = classify_question(r, g)
        if diag["is_correct"]:
            correct_items.append(diag)
        else:
            failed_items.append(diag)

    summary_counts = {
        "Retrieval failures": sum(1 for x in failed_items if x["failure_type"] == "Retrieval failure"),
        "Context failures": sum(1 for x in failed_items if x["failure_type"] == "Context insufficient"),
        "LLM failures": sum(1 for x in failed_items if x["failure_type"] == "LLM misunderstood context"),
        "Hallucinations": sum(1 for x in failed_items if x["failure_type"] == "Hallucination / unsupported claim"),
        "Ambiguous/difficult": sum(1 for x in failed_items if x["failure_type"] == "Difficult / ambiguous question"),
        "Other": 0
    }

    return {
        "total_questions": len(results),
        "correct_count": len(correct_items),
        "incorrect_count": len(failed_items),
        "correct_items": correct_items,
        "failed_items": failed_items,
        "summary_counts": summary_counts
    }


def generate_markdown(data: Dict[str, Any], output_path: Path):
    lines = []
    lines.append(f"{data['total_questions']} total questions")
    lines.append("│")
    lines.append(f"├── Correct answers ({data['correct_count']})")
    lines.append("│")
    lines.append(f"└── Incorrect answers ({data['incorrect_count']})")
    lines.append("      │")
    sc = data["summary_counts"]
    lines.append(f"      ├── Retrieval failure ({sc['Retrieval failures']})")
    lines.append(f"      ├── Context insufficient ({sc['Context failures']})")
    lines.append(f"      ├── LLM misunderstood context ({sc['LLM failures']})")
    lines.append(f"      ├── Hallucination / unsupported claim ({sc['Hallucinations']})")
    lines.append(f"      └── Difficult / ambiguous question ({sc['Ambiguous/difficult']})\n")

    lines.append("## Failed Questions Breakdown\n")
    lines.append("| Question ID | Question | Golden Answer | Generated Answer | Top-5 Retrieval | Evidence Found? | Failure Type | Reason |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- |")

    for f in data["failed_items"]:
        clean_q = f["question"].replace("|", "\\|").replace("\n", " ")
        clean_g = f["golden_answer"].replace("|", "\\|").replace("\n", " ")
        clean_a = f["generated_answer"].replace("|", "\\|").replace("\n", " ")
        clean_r = f["reason"].replace("|", "\\|").replace("\n", " ")
        clean_top = f["top5_retrieval"].replace("|", "\\|")

        lines.append(f"| `{f['qid']}` | {clean_q} | {clean_g} | {clean_a} | {clean_top} | {f['evidence_found']} | **{f['failure_type']}** | {clean_r} |")

    lines.append("\n```text")
    lines.append("Failure Analysis Summary")
    lines.append("────────────────────────")
    lines.append(f"Retrieval failures:       {sc['Retrieval failures']}")
    lines.append(f"Context failures:         {sc['Context failures']}")
    lines.append(f"LLM failures:             {sc['LLM failures']}")
    lines.append(f"Hallucinations:           {sc['Hallucinations']}")
    lines.append(f"Ambiguous/difficult:      {sc['Ambiguous/difficult']}")
    lines.append(f"Other:                    {sc['Other']}")
    lines.append("```\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Structured RAG Failure Taxonomy Analyzer")
    parser.add_argument("--file", type=str, default=str(DEFAULT_RESULTS), help="Path to evaluation results JSON")
    args = parser.parse_args()

    results_file = Path(args.file)
    data = analyze_all(results_file)

    # Save Markdown
    generate_markdown(data, OUTPUT_MD)

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    sc = data["summary_counts"]
    print(f"{data['total_questions']} total questions")
    print("│")
    print(f"├── Correct answers ({data['correct_count']})")
    print("│")
    print(f"└── Incorrect answers ({data['incorrect_count']})")
    print("      │")
    print(f"      ├── Retrieval failure ({sc['Retrieval failures']})")
    print(f"      ├── Context insufficient ({sc['Context failures']})")
    print(f"      ├── LLM misunderstood context ({sc['LLM failures']})")
    print(f"      ├── Hallucination / unsupported claim ({sc['Hallucinations']})")
    print(f"      └── Difficult / ambiguous question ({sc['Ambiguous/difficult']})")
    print()
    print("Failure Analysis Summary")
    print("────────────────────────")
    print(f"Retrieval failures:       {sc['Retrieval failures']}")
    print(f"Context failures:         {sc['Context failures']}")
    print(f"LLM failures:             {sc['LLM failures']}")
    print(f"Hallucinations:           {sc['Hallucinations']}")
    print(f"Ambiguous/difficult:      {sc['Ambiguous/difficult']}")
    print(f"Other:                    {sc['Other']}")
    print(f"\nSaved report to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()

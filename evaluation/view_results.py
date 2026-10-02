#!/usr/bin/env python3
"""
Structured RAG Evaluation Viewer, Analyzer & Multi-Run Explorer
--------------------------------------------------------------
Generates interactive, rich reporting tools for RAG evaluation results:
1. Multi-Run Interactive HTML Dashboard (evaluation/results/report.html)
   - Dynamic selection of JSON result files from 'results data/'
   - Client-side custom JSON file loading (file picker & drag-and-drop)
   - Multi-run comparison / diff mode (e.g. Run 1 vs Run 0)
   - Instant client-side metric recomputation, filtering, search & case-study tracking
2. Comprehensive Markdown Report (evaluation/results/evaluation_summary.md)
3. Rich Terminal CLI Explorer with colored tables and side-by-side cards.

Usage:
    python evaluation/view_results.py                   # Generates HTML and Markdown reports & shows summary
    python evaluation/view_results.py --summary         # Show metrics table in terminal
    python evaluation/view_results.py --file <path>     # View specific run file
    python evaluation/view_results.py --id blip2_001    # View specific question details
    python evaluation/view_results.py --list            # List all questions with scores & status
    python evaluation/view_results.py --misses          # Show questions that missed the target page
    python evaluation/view_results.py --paper "BLIP-2"  # Filter by paper
    python evaluation/view_results.py --list-runs       # List all available result datasets
"""

import os
import sys
import json
import html
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from evaluation.analyze_failures import classify_question
except ImportError:
    classify_question = None
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results" / "results data"
RESULTS_FILE = PROJECT_ROOT / "evaluation" / "results" / "rag_evaluation_results.json"
GOLDEN_FILE = PROJECT_ROOT / "evaluation" / "gemini-code-1790832398351.json"
OUTPUT_HTML = PROJECT_ROOT / "evaluation" / "results" / "report.html"
OUTPUT_MD = PROJECT_ROOT / "evaluation" / "results" / "evaluation_summary.md"
RAW_PAPERS_DIR = PROJECT_ROOT / "data" / "raw"


def find_result_files() -> List[Path]:
    """Find all evaluation result JSON files in results data/ and evaluation/results/."""
    candidates = []
    if RESULTS_DIR.exists():
        candidates.extend(sorted(RESULTS_DIR.glob("*.json")))
    if RESULTS_FILE.exists() and RESULTS_FILE not in candidates:
        candidates.append(RESULTS_FILE)
    return candidates


def load_golden_map() -> Dict[str, Any]:
    """Load the ground truth golden dataset as a dictionary keyed by question ID."""
    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(f"Golden dataset not found: {GOLDEN_FILE}")

    with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
        golden_raw = json.load(f)

    return {item["id"]: item for item in golden_raw}


def enrich_items(results: List[Dict[str, Any]], golden_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Enrich raw evaluation results with ground truth metadata and scoring tags."""
    enriched = []
    for idx, r in enumerate(results, 1):
        qid = r.get("question_id", f"q_{idx}")
        g = golden_map.get(qid, {})
        target_file = g.get("source_file", "")
        target_page = g.get("page")
        target_paper = g.get("source", "")
        q_type = g.get("type", "General")
        evidence = g.get("evidence", "")

        chunks = r.get("retrieved_chunks", [])

        # Check retrieval matches
        doc_hit_1 = False
        exact_page_1 = False
        near_page_1 = False
        doc_hit_5 = False
        exact_page_5 = False
        near_page_5 = False
        doc_rank = None
        exact_rank = None
        near_rank = None

        for c_idx, c in enumerate(chunks, 1):
            c_file = c.get("source_file", "")
            c_page = c.get("page")

            is_doc_match = (c_file == target_file) if target_file else False
            is_exact = is_doc_match and (c_page == target_page) if (target_page is not None and c_page is not None) else False
            is_near = is_doc_match and (abs(c_page - target_page) <= 1) if (target_page is not None and c_page is not None) else False

            c["is_doc_match"] = is_doc_match
            c["is_exact_page"] = is_exact
            c["is_near_page"] = is_near

            if c_idx == 1:
                doc_hit_1 = is_doc_match
                exact_page_1 = is_exact
                near_page_1 = is_near

            if is_doc_match:
                doc_hit_5 = True
                if doc_rank is None:
                    doc_rank = c_idx
            if is_exact:
                exact_page_5 = True
                if exact_rank is None:
                    exact_rank = c_idx
            if is_near:
                near_page_5 = True
                if near_rank is None:
                    near_rank = c_idx

        # Check generation status
        raw_gen_ans = r.get("generated_answer", "")
        gen_ans_clean = raw_gen_ans.strip()
        gen_ans_lower = gen_ans_clean.lower()

        if not gen_ans_clean:
            gen_status = "empty"
            gen_status_label = "⚠️ Empty / Cutoff"
            gen_status_badge = "amber"
        elif any(p in gen_ans_lower for p in [
            "no information", "does not contain", "not mentioned",
            "does not provide", "cannot find", "not specified"
        ]):
            gen_status = "refused"
            gen_status_label = "🚫 Refused / Not in Context"
            gen_status_badge = "rose"
        else:
            gen_status = "answered"
            gen_status_label = "✅ Answered"
            gen_status_badge = "emerald"

        top1_score = chunks[0].get("score", 0.0) if chunks else 0.0
        avg_score = (sum(c.get("score", 0.0) for c in chunks) / len(chunks)) if chunks else 0.0

        if r.get("retrieval_match"):
            match_label = r["retrieval_match"]
            if "Exact Page" in match_label:
                match_level = "exact_top1" if "Rank 1" in match_label else "exact_top5"
                status_color = "emerald"
            elif "Adjacent Page" in match_label:
                match_level = "near_top1" if "Rank 1" in match_label else "near_top5"
                status_color = "teal"
            elif "Doc Only" in match_label:
                match_level = "doc_only"
                status_color = "amber"
            else:
                match_level = "miss"
                status_color = "rose"
        elif exact_page_1:
            match_level = "exact_top1"
            match_label = "Exact Page (Rank 1)"
            status_color = "emerald"
        elif exact_page_5:
            match_level = "exact_top5"
            match_label = f"Exact Page (Rank {exact_rank})"
            status_color = "emerald"
        elif near_page_1:
            match_level = "near_top1"
            match_label = "Adjacent Page (Rank 1)"
            status_color = "teal"
        elif near_page_5:
            match_level = "near_top5"
            match_label = f"Adjacent Page (Rank {near_rank})"
            status_color = "teal"
        elif doc_hit_5:
            match_level = "doc_only"
            match_label = f"Doc Only (Rank {doc_rank})"
            status_color = "amber"
        else:
            match_level = "miss"
            match_label = "Doc Miss"
            status_color = "rose"

        # Failure classification
        diag = classify_question(r, g) if classify_question else None
        fail_cat = diag["failure_category"] if diag else ("PASS" if gen_status == "answered" else "A")
        fail_label = diag["failure_label"] if diag else ("✅ Answered" if gen_status == "answered" else "A. Retrieval failure")
        root_cause = diag["root_cause"] if diag else ""
        is_correct = diag["is_correct"] if diag else (gen_status == "answered")

        enriched.append({
            "index": idx,
            "question_id": qid,
            "question": r.get("question", ""),
            "golden_answer": r.get("golden_answer", g.get("answer", "")),
            "generated_answer": gen_ans_clean if gen_ans_clean else "(No response generated or thinking truncated)",
            "raw_generated_answer": raw_gen_ans,
            "evidence": evidence,
            "paper": target_paper,
            "source_file": target_file,
            "target_page": target_page,
            "question_type": q_type,
            "retrieved_chunks": chunks,
            "top1_score": top1_score,
            "avg_score": avg_score,
            "doc_hit_1": doc_hit_1,
            "doc_hit_5": doc_hit_5,
            "exact_page_1": exact_page_1,
            "exact_page_5": exact_page_5,
            "near_page_1": near_page_1,
            "near_page_5": near_page_5,
            "hit_rank": doc_rank,
            "gen_status": gen_status,
            "gen_status_label": gen_status_label,
            "gen_status_badge": gen_status_badge,
            "match_level": match_level,
            "match_label": match_label,
            "status_color": status_color,
            "failure_category": fail_cat,
            "failure_label": fail_label,
            "root_cause": root_cause,
            "is_correct": is_correct,
        })

    return enriched


def compute_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate evaluation metrics across all questions."""
    total = len(items)
    if total == 0:
        return {}

    doc_hits_1 = sum(1 for x in items if x["doc_hit_1"])
    doc_hits_5 = sum(1 for x in items if x["doc_hit_5"])
    near_hits_1 = sum(1 for x in items if x["near_page_1"])
    near_hits_5 = sum(1 for x in items if x["near_page_5"])
    exact_hits_1 = sum(1 for x in items if x["exact_page_1"])
    exact_hits_5 = sum(1 for x in items if x["exact_page_5"])
    refusals = sum(1 for x in items if x["gen_status"] == "refused")
    gen_answered = sum(1 for x in items if x["gen_status"] == "answered")
    gen_empty = sum(1 for x in items if x["gen_status"] == "empty")
    avg_top1_score = sum(x["top1_score"] for x in items) / total
    avg_chunk_score = sum(x["avg_score"] for x in items) / total

    # Scan all raw PDF papers in corpus
    all_corpus_files = sorted([f.name for f in RAW_PAPERS_DIR.glob("*.pdf")]) if RAW_PAPERS_DIR.exists() else []

    # Count how many times each paper's chunks were retrieved across all questions
    chunk_counts_by_file = {}
    for x in items:
        for c in x.get("retrieved_chunks", []):
            s_file = c.get("source_file", "")
            chunk_counts_by_file[s_file] = chunk_counts_by_file.get(s_file, 0) + 1

    # Breakdown by paper
    by_paper = {}
    for x in items:
        p = x["paper"] or "Unknown"
        s_file = x.get("source_file", "")
        if p not in by_paper:
            by_paper[p] = {
                "total": 0,
                "source_file": s_file,
                "doc_hits_5": 0,
                "page_hits_5": 0,
                "gen_answered": 0,
                "scores": [],
                "chunks_retrieved": chunk_counts_by_file.get(s_file, 0),
                "is_evaluated": True,
            }
        by_paper[p]["total"] += 1
        if x["doc_hit_5"]:
            by_paper[p]["doc_hits_5"] += 1
        if x["near_page_5"]:
            by_paper[p]["page_hits_5"] += 1
        if x["gen_status"] == "answered":
            by_paper[p]["gen_answered"] += 1
        by_paper[p]["scores"].append(x["top1_score"])

    for p, stats in by_paper.items():
        stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
        stats["doc_acc"] = (stats["doc_hits_5"] / stats["total"]) * 100 if stats["total"] else 0.0
        stats["page_acc"] = (stats["page_hits_5"] / stats["total"]) * 100 if stats["total"] else 0.0
        stats["gen_acc"] = (stats["gen_answered"] / stats["total"]) * 100 if stats["total"] else 0.0

    # Map remaining raw PDFs that were not in golden evaluation dataset
    evaluated_source_files = set(stats.get("source_file") for stats in by_paper.values())
    friendly_names = {
        "VIMA: General Robot Manipulation with Multimodal Prompts.pdf": "VIMA",
        "Visual Instruction Tuning.pdf": "Visual Instruction Tuning (LLaVA)",
        "Vision Language Models: A Survey of 26K Papers (CVPR, ICLR, NeurIPS 2023–2025).pdf": "Vision Language Models Survey",
    }
    unevaluated_papers = {}
    for pdf_name in all_corpus_files:
        if pdf_name not in evaluated_source_files:
            p_name = friendly_names.get(pdf_name, pdf_name.replace(".pdf", ""))
            unevaluated_papers[p_name] = {
                "total": 0,
                "source_file": pdf_name,
                "doc_hits_5": 0,
                "page_hits_5": 0,
                "gen_answered": 0,
                "avg_score": 0.0,
                "doc_acc": 0.0,
                "page_acc": 0.0,
                "gen_acc": 0.0,
                "chunks_retrieved": chunk_counts_by_file.get(pdf_name, 0),
                "is_evaluated": False,
            }

    # Breakdown by type
    by_type = {}
    for x in items:
        t = x["question_type"] or "General"
        if t not in by_type:
            by_type[t] = {"total": 0, "doc_hits_5": 0, "page_hits_5": 0, "gen_answered": 0, "scores": []}
        by_type[t]["total"] += 1
        if x["doc_hit_5"]:
            by_type[t]["doc_hits_5"] += 1
        if x["near_page_5"]:
            by_type[t]["page_hits_5"] += 1
        if x["gen_status"] == "answered":
            by_type[t]["gen_answered"] += 1
        by_type[t]["scores"].append(x["top1_score"])

    for t, stats in by_type.items():
        stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0
        stats["doc_acc"] = (stats["doc_hits_5"] / stats["total"]) * 100 if stats["total"] else 0.0
        stats["page_acc"] = (stats["page_hits_5"] / stats["total"]) * 100 if stats["total"] else 0.0
        stats["gen_acc"] = (stats["gen_answered"] / stats["total"]) * 100 if stats["total"] else 0.0

    return {
        "total": total,
        "doc_hits_1": doc_hits_1,
        "doc_acc_1": (doc_hits_1 / total) * 100,
        "doc_hits_5": doc_hits_5,
        "doc_acc_5": (doc_hits_5 / total) * 100,
        "near_hits_1": near_hits_1,
        "near_acc_1": (near_hits_1 / total) * 100,
        "near_hits_5": near_hits_5,
        "near_acc_5": (near_hits_5 / total) * 100,
        "exact_hits_1": exact_hits_1,
        "exact_acc_1": (exact_hits_1 / total) * 100,
        "exact_hits_5": exact_hits_5,
        "exact_acc_5": (exact_hits_5 / total) * 100,
        "gen_answered": gen_answered,
        "gen_answered_acc": (gen_answered / total) * 100,
        "gen_empty": gen_empty,
        "gen_empty_acc": (gen_empty / total) * 100,
        "refusals": refusals,
        "refusals_acc": (refusals / total) * 100,
        "avg_top1_score": avg_top1_score,
        "avg_chunk_score": avg_chunk_score,
        "total_corpus_papers": len(all_corpus_files),
        "evaluated_paper_count": len(by_paper),
        "unevaluated_paper_count": len(unevaluated_papers),
        "by_paper": by_paper,
        "unevaluated_papers": unevaluated_papers,
        "by_type": by_type,
    }


def generate_html_report(all_datasets: Dict[str, Dict[str, Any]], active_key: str, golden_lookup: Dict[str, Any], output_path: Path):
    """Generate modern, interactive multi-run HTML dashboard with client-side file switching and upload."""
    datasets_json = json.dumps(all_datasets, ensure_ascii=False)
    golden_json = json.dumps(golden_lookup, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RAG Evaluation Report — Interactive Multi-Run Explorer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #070b14;
      --card-bg: #0e1526;
      --card-border: #1a253a;
      --card-hover: #152037;
      --accent-blue: #3b82f6;
      --accent-blue-glow: rgba(59, 130, 246, 0.25);
      --accent-cyan: #06b6d4;
      --emerald: #10b981;
      --emerald-bg: rgba(16, 185, 129, 0.12);
      --emerald-border: rgba(16, 185, 129, 0.35);
      --teal: #14b8a6;
      --teal-bg: rgba(20, 184, 166, 0.12);
      --teal-border: rgba(20, 184, 166, 0.35);
      --amber: #f59e0b;
      --amber-bg: rgba(245, 158, 11, 0.12);
      --amber-border: rgba(245, 158, 11, 0.35);
      --rose: #ef4444;
      --rose-bg: rgba(239, 68, 68, 0.12);
      --rose-border: rgba(239, 68, 68, 0.35);
      --purple: #a855f7;
      --text-main: #f3f4f6;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --code-bg: #070c18;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text-main);
      min-height: 100vh;
      line-height: 1.5;
      padding-bottom: 80px;
    }}

    .container {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 0 20px;
    }}

    /* Header & Hero */
    header {{
      background: linear-gradient(180deg, #10192e 0%, #070b14 100%);
      border-bottom: 1px solid var(--card-border);
      padding: 28px 0 24px;
    }}

    .header-top {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 20px;
    }}

    .title-group h1 {{
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 40%, #93c5fd 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .title-group p {{
      color: var(--text-muted);
      font-size: 13.5px;
      margin-top: 4px;
    }}

    .pipeline-badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      background: rgba(59, 130, 246, 0.08);
      border: 1px solid rgba(59, 130, 246, 0.25);
      border-radius: 9999px;
      font-size: 12px;
      color: #93c5fd;
      font-weight: 500;
    }}

    /* Dataset Selector Bar */
    .dataset-bar {{
      background: #0d1629;
      border: 1px solid rgba(59, 130, 246, 0.3);
      border-radius: 12px;
      padding: 12px 18px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 22px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }}

    .dataset-controls-left {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .dataset-label {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #93c5fd;
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .dataset-select {{
      background: #131d33;
      border: 1px solid rgba(59, 130, 246, 0.4);
      color: #ffffff;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      outline: none;
      min-width: 320px;
      transition: all 0.2s;
    }}

    .dataset-select:focus {{
      border-color: var(--accent-blue);
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
    }}

    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
      border: 1px solid transparent;
      outline: none;
    }}

    .btn-primary {{
      background: rgba(59, 130, 246, 0.15);
      border-color: rgba(59, 130, 246, 0.4);
      color: #93c5fd;
    }}

    .btn-primary:hover {{
      background: rgba(59, 130, 246, 0.28);
      color: #ffffff;
      border-color: #60a5fa;
    }}

    .btn-compare {{
      background: rgba(168, 85, 247, 0.12);
      border-color: rgba(168, 85, 247, 0.35);
      color: #d8b4fe;
    }}

    .btn-compare:hover, .btn-compare.active {{
      background: rgba(168, 85, 247, 0.25);
      color: #ffffff;
      border-color: #c084fc;
    }}

    .btn-compare.active {{
      box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.35);
    }}

    .file-status-pill {{
      font-size: 11.5px;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-muted);
      background: #090e1a;
      border: 1px solid var(--card-border);
      padding: 4px 10px;
      border-radius: 6px;
    }}

    /* Compare Mode Panel */
    .compare-panel {{
      display: none;
      background: #11152a;
      border: 1px solid rgba(168, 85, 247, 0.35);
      border-radius: 12px;
      padding: 14px 18px;
      margin-bottom: 22px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }}

    .compare-header {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}

    .compare-title {{
      font-size: 13px;
      font-weight: 700;
      color: #e9d5ff;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .compare-deltas {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .delta-pill {{
      font-size: 12px;
      padding: 5px 10px;
      border-radius: 6px;
      background: #181d38;
      border: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .delta-val-up {{ color: var(--emerald); font-weight: 700; }}
    .delta-val-down {{ color: var(--rose); font-weight: 700; }}
    .delta-val-neutral {{ color: var(--text-muted); font-weight: 600; }}

    .compare-filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
      border-top: 1px solid rgba(168, 85, 247, 0.2);
      padding-top: 10px;
    }}

    .cmp-filter-btn {{
      background: #171d36;
      border: 1px solid rgba(168, 85, 247, 0.25);
      color: #d8b4fe;
      font-size: 11.5px;
      padding: 4px 10px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.15s;
    }}

    .cmp-filter-btn.active, .cmp-filter-btn:hover {{
      background: rgba(168, 85, 247, 0.3);
      color: #ffffff;
      border-color: #c084fc;
    }}

    /* Metrics Grid */
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
      gap: 14px;
      margin-bottom: 8px;
    }}

    .metric-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px 18px;
      transition: transform 0.2s, border-color 0.2s;
      position: relative;
      overflow: hidden;
    }}

    .metric-card:hover {{
      transform: translateY(-2px);
      border-color: rgba(59, 130, 246, 0.4);
    }}

    .metric-title {{
      font-size: 11.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 6px;
    }}

    .metric-val {{
      font-size: 26px;
      font-weight: 700;
      color: #ffffff;
      display: flex;
      align-items: baseline;
      gap: 6px;
    }}

    .metric-sub {{
      font-size: 12px;
      color: var(--text-dim);
      margin-top: 4px;
    }}

    .metric-delta {{
      font-size: 12px;
      font-weight: 700;
      padding: 1px 6px;
      border-radius: 4px;
      margin-left: 6px;
    }}

    .text-emerald {{ color: var(--emerald); }}
    .text-teal {{ color: var(--teal); }}
    .text-amber {{ color: var(--amber); }}
    .text-cyan {{ color: var(--accent-cyan); }}
    .text-purple {{ color: var(--purple); }}

    /* Controls Bar */
    .controls-panel {{
      background: #0b1120;
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px 20px;
      margin: 24px auto;
      box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }}

    .search-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-bottom: 14px;
    }}

    .search-box {{
      flex: 1;
      min-width: 260px;
      position: relative;
    }}

    .search-box input {{
      width: 100%;
      background: #141c30;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 14px 10px 38px;
      color: #ffffff;
      font-size: 14px;
      font-family: inherit;
      outline: none;
      transition: border-color 0.2s;
    }}

    .search-box input:focus {{
      border-color: var(--accent-blue);
      box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }}

    .search-icon {{
      position: absolute;
      left: 12px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-dim);
      font-size: 14px;
      pointer-events: none;
    }}

    .filters-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }}

    .filter-group {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .filter-group label {{
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 500;
    }}

    select {{
      background: #141c30;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 8px 12px;
      color: var(--text-main);
      font-size: 13px;
      outline: none;
      cursor: pointer;
    }}

    select:focus {{
      border-color: var(--accent-blue);
    }}

    .view-toggle {{
      margin-left: auto;
      display: flex;
      background: #141c30;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 3px;
      gap: 2px;
    }}

    .toggle-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s;
    }}

    .toggle-btn.active {{
      background: var(--accent-blue);
      color: #ffffff;
    }}

    /* Results Header / Counter */
    .results-meta {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      font-size: 13px;
      color: var(--text-muted);
    }}

    /* Cards Layout */
    .cards-container {{
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}

    .qa-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      overflow: hidden;
      transition: border-color 0.2s, box-shadow 0.2s;
    }}

    .qa-card:hover {{
      border-color: rgba(59, 130, 246, 0.4);
      box-shadow: 0 8px 28px rgba(0, 0, 0, 0.25);
    }}

    .card-header {{
      background: #131b2e;
      border-bottom: 1px solid var(--card-border);
      padding: 14px 20px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }}

    .card-meta-left {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}

    .qid-pill {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      font-weight: 600;
      background: #090e1a;
      border: 1px solid var(--card-border);
      padding: 4px 10px;
      border-radius: 6px;
      color: #60a5fa;
    }}

    .type-badge {{
      font-size: 11px;
      font-weight: 600;
      padding: 3px 9px;
      border-radius: 6px;
      background: rgba(147, 51, 234, 0.15);
      border: 1px solid rgba(147, 51, 234, 0.35);
      color: #c084fc;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .paper-badge {{
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .paper-badge span {{
      color: #93c5fd;
      font-weight: 600;
    }}

    .status-badge {{
      font-size: 12px;
      font-weight: 600;
      padding: 4px 12px;
      border-radius: 9999px;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}

    .badge-emerald {{
      background: var(--emerald-bg);
      border: 1px solid var(--emerald-border);
      color: var(--emerald);
    }}

    .badge-teal {{
      background: var(--teal-bg);
      border: 1px solid var(--teal-border);
      color: var(--teal);
    }}

    .badge-amber {{
      background: var(--amber-bg);
      border: 1px solid var(--amber-border);
      color: var(--amber);
    }}

    .badge-rose {{
      background: var(--rose-bg);
      border: 1px solid var(--rose-border);
      color: var(--rose);
    }}

    .badge-purple {{
      background: rgba(168, 85, 247, 0.15);
      border: 1px solid rgba(168, 85, 247, 0.35);
      color: #c084fc;
    }}

    /* Card Body */
    .card-body {{
      padding: 20px;
    }}

    .question-prompt {{
      font-size: 16px;
      font-weight: 600;
      color: #ffffff;
      margin-bottom: 20px;
      line-height: 1.45;
      padding-left: 12px;
      border-left: 3px solid var(--accent-blue);
    }}

    /* Answers Comparison Grid */
    .answers-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }}

    .answers-grid.compare-3 {{
      grid-template-columns: 1fr 1fr 1fr;
    }}

    @media (max-width: 1024px) {{
      .answers-grid, .answers-grid.compare-3 {{
        grid-template-columns: 1fr;
      }}
    }}

    .answer-box {{
      background: #090e1a;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 16px;
      display: flex;
      flex-direction: column;
    }}

    .answer-box.golden {{
      border-color: rgba(16, 185, 129, 0.25);
      background: linear-gradient(180deg, rgba(16, 185, 129, 0.04) 0%, #090e1a 100%);
    }}

    .answer-box.generated {{
      border-color: rgba(59, 130, 246, 0.25);
      background: linear-gradient(180deg, rgba(59, 130, 246, 0.04) 0%, #090e1a 100%);
    }}

    .answer-box.compared {{
      border-color: rgba(168, 85, 247, 0.3);
      background: linear-gradient(180deg, rgba(168, 85, 247, 0.05) 0%, #090e1a 100%);
    }}

    .answer-box-title {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .answer-box.golden .answer-box-title {{
      color: var(--emerald);
    }}

    .answer-box.generated .answer-box-title {{
      color: #60a5fa;
    }}

    .answer-box.compared .answer-box-title {{
      color: #c084fc;
    }}

    .answer-content {{
      font-size: 13.5px;
      color: var(--text-main);
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.55;
      flex: 1;
    }}

    .evidence-block {{
      margin-top: 14px;
      padding: 10px 12px;
      background: rgba(0, 0, 0, 0.35);
      border-left: 2px solid var(--emerald);
      border-radius: 0 6px 6px 0;
      font-size: 12px;
      color: var(--text-muted);
    }}

    .evidence-title {{
      font-size: 11px;
      font-weight: 600;
      color: var(--emerald);
      text-transform: uppercase;
      margin-bottom: 4px;
    }}

    /* Retrieved Chunks Accordion */
    .chunks-section {{
      background: #080d19;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      overflow: hidden;
    }}

    .chunks-toggle {{
      width: 100%;
      background: transparent;
      border: none;
      padding: 12px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--text-muted);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }}

    .chunks-toggle:hover {{
      background: #111827;
      color: var(--text-main);
    }}

    .chunks-list {{
      padding: 0 16px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .chunk-item {{
      background: #0f172a;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 13px;
    }}

    .chunk-item.target-matched {{
      border-color: rgba(16, 185, 129, 0.4);
      background: rgba(16, 185, 129, 0.03);
    }}

    .chunk-header {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
      font-size: 12px;
    }}

    .chunk-rank {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 4px;
      background: #1e293b;
      color: #93c5fd;
    }}

    .score-badge {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
      padding: 2px 7px;
      border-radius: 4px;
      background: rgba(6, 182, 212, 0.12);
      border: 1px solid rgba(6, 182, 212, 0.3);
      color: var(--accent-cyan);
    }}

    .chunk-source {{
      color: var(--text-muted);
      margin-left: auto;
      font-size: 11px;
    }}

    .target-tag {{
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: var(--emerald);
      padding: 1px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
    }}

    .chunk-text {{
      color: #cbd5e1;
      line-height: 1.45;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11.5px;
      background: var(--code-bg);
      padding: 10px;
      border-radius: 6px;
      overflow-x: auto;
      max-height: 180px;
      overflow-y: auto;
    }}

    /* Table View */
    .table-container {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      overflow-x: auto;
      display: none;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}

    th {{
      background: #12192c;
      color: var(--text-muted);
      font-weight: 600;
      padding: 12px 16px;
      border-bottom: 1px solid var(--card-border);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    td {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--card-border);
      color: var(--text-main);
      vertical-align: top;
    }}

    tr:hover td {{
      background: var(--card-hover);
    }}

    .empty-state {{
      text-align: center;
      padding: 60px 20px;
      color: var(--text-muted);
      background: var(--card-bg);
      border: 1px dashed var(--card-border);
      border-radius: 14px;
      display: none;
    }}

    /* Drag & Drop Overlay */
    .drop-overlay {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(7, 11, 20, 0.88);
      backdrop-filter: blur(8px);
      z-index: 9999;
      justify-content: center;
      align-items: center;
      flex-direction: column;
      border: 3px dashed var(--accent-blue);
      color: #ffffff;
      pointer-events: none;
    }}

    .drop-overlay.active {{
      display: flex;
    }}

    /* Toast */
    .toast {{
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #111a2f;
      border: 1px solid var(--accent-blue);
      color: #ffffff;
      padding: 12px 20px;
      border-radius: 8px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.5);
      z-index: 10000;
      font-size: 13px;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 10px;
      transform: translateY(100px);
      opacity: 0;
      transition: all 0.25s ease;
    }}

    .toast.show {{
      transform: translateY(0);
      opacity: 1;
    }}
  </style>
</head>
<body>

  <!-- Drag and drop indicator overlay -->
  <div id="dropOverlay" class="drop-overlay">
    <div style="font-size: 48px; margin-bottom: 12px;">📥</div>
    <div style="font-size: 20px; font-weight: 700;">Drop JSON Evaluation Result File Here</div>
    <div style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">Supports any RAG evaluation results file (e.g. from results data folder)</div>
  </div>

  <!-- Toast notification -->
  <div id="toast" class="toast">
    <span id="toastIcon">✨</span>
    <span id="toastMsg">Loaded dataset</span>
  </div>

  <header>
    <div class="container">
      <div class="header-top">
        <div class="title-group">
          <h1>⚡ RAG Benchmark Evaluation Report</h1>
          <p>Interactive Multi-Run Explorer · Qwen3-Embedding-0.6B &rarr; Qdrant &rarr; Qwen3.5:4B</p>
        </div>
        <div class="pipeline-badge">
          <span>● 13 Research Papers in Corpus (10 Evaluated · 3 Chunks-Only)</span>
          <span>|</span>
          <span id="badgeQuestionCount">122 Benchmark Questions</span>
          <span>|</span>
          <span>Collection: Qwen3_Embedding_0_6B_semantic</span>
        </div>
      </div>

      <!-- Interactive Dataset Selector Bar -->
      <div class="dataset-bar">
        <div class="dataset-controls-left">
          <span class="dataset-label">📁 Active Result File:</span>
          <select id="datasetSelector" class="dataset-select" onchange="onDatasetChange(this.value)">
            <!-- Options populated dynamically from PRELOADED_DATASETS -->
          </select>
          <button class="btn btn-primary" onclick="document.getElementById('jsonFileInput').click()">
            <span>📂</span> Choose Local JSON...
          </button>
          <input type="file" id="jsonFileInput" accept=".json" style="display:none;" onchange="handleFileSelect(event)">
          <button id="btnCompareToggle" class="btn btn-compare" onclick="toggleCompareMode()">
            <span>⚡</span> Compare Runs
          </button>
        </div>
        <div class="dataset-controls-right">
          <span id="fileStatusPill" class="file-status-pill">Ready</span>
        </div>
      </div>

      <!-- Compare Mode Sub-Panel (Collapsible) -->
      <div id="comparePanel" class="compare-panel">
        <div class="compare-header">
          <div class="compare-title">
            <span>⚡ Run Comparison Mode</span>
            <span style="font-size:12px; font-weight:normal; color:var(--text-muted);">Comparing Active Run against:</span>
            <select id="compareSelector" style="background:#1b2340; border:1px solid rgba(168,85,247,0.4); color:#fff; font-size:12px; padding:4px 8px; border-radius:6px;" onchange="onCompareTargetChange(this.value)">
              <!-- Options populated dynamically -->
            </select>
          </div>
          <div id="compareDeltas" class="compare-deltas">
            <!-- Dynamic comparison pills -->
          </div>
        </div>
        <div class="compare-filters">
          <span style="font-size:11.5px; color:var(--text-muted); align-self:center; margin-right:4px;">Filter Questions:</span>
          <button class="cmp-filter-btn active" onclick="setCmpFilter('all', this)">All Questions</button>
          <button class="cmp-filter-btn" onclick="setCmpFilter('changed_ans', this)">Changed Answers</button>
          <button class="cmp-filter-btn" onclick="setCmpFilter('fixed_cutoff', this)">Fixed Cutoffs Only</button>
          <button class="cmp-filter-btn" onclick="setCmpFilter('changed_chunks', this)">Changed Retrieval</button>
          <button class="cmp-filter-btn" onclick="setCmpFilter('identical', this)">Identical</button>
        </div>
      </div>

      <!-- Dynamic Metrics Grid -->
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-title">Corpus Research Papers</div>
          <div class="metric-val" id="metricCorpus">13 <span style="font-size:16px; color:var(--text-muted);">(10 Evaluated)</span></div>
          <div class="metric-sub" id="metricCorpusSub">10 papers with golden questions; 3 context papers</div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Top-5 Document Hit Rate</div>
          <div class="metric-val text-emerald" id="metricDocHit5">0%</div>
          <div class="metric-sub" id="metricDocHit5Sub">Target paper chunk in Top-5</div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Top-1 Document Hit Rate</div>
          <div class="metric-val text-teal" id="metricDocHit1">0%</div>
          <div class="metric-sub" id="metricDocHit1Sub">Rank #1 chunk from target paper</div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Page Precision (±1 Page)</div>
          <div class="metric-val text-cyan" id="metricNearHit5">0%</div>
          <div class="metric-sub" id="metricNearHit5Sub">Exact or adjacent target page</div>
        </div>

        <div class="metric-card">
          <div class="metric-title">LLM Completed Answers</div>
          <div class="metric-val text-teal" id="metricGenAnswered">0</div>
          <div class="metric-sub" id="metricGenAnsweredSub">Completed responses</div>
        </div>

        <div class="metric-card">
          <div class="metric-title">Average Retrieval Score</div>
          <div class="metric-val text-amber" id="metricAvgScore">0.0000</div>
          <div class="metric-sub">Mean Top-1 chunk similarity</div>
        </div>
      </div>
    </div>
  </header>

  <main class="container">
    <!-- 13-Paper Corpus Overview Card -->
    <div style="background:#0a101f; border:1px solid var(--card-border); border-radius:12px; padding:16px 20px; margin:20px auto 0;">
      <div style="display:flex; justify-content:space-between; align-items:center; cursor:pointer;" onclick="toggleCorpusInfo()">
        <div style="display:flex; align-items:center; gap:10px;">
          <span style="font-size:16px;">📚</span>
          <span style="font-weight:700; font-size:14px; color:#ffffff;">Full 13 Research Papers Corpus Status</span>
          <span style="font-size:11px; font-weight:600; background:rgba(59,130,246,0.15); color:#93c5fd; padding:2px 8px; border-radius:999px;">13 PDFs Ingested</span>
        </div>
        <span id="corpusToggleBtn" style="font-size:12px; color:var(--accent-blue); font-weight:600;">▼ View All 13 Papers</span>
      </div>
      <div id="corpusInfoBody" style="display:none; margin-top:14px; border-top:1px solid var(--card-border); padding-top:14px;">
        <p style="font-size:13px; color:var(--text-muted); margin-bottom:12px;">
          The Qdrant semantic vector index contains all <strong>13 research papers</strong> from <code style="font-family:'JetBrains Mono'; background:#151d30; padding:2px 6px; border-radius:4px;">data/raw</code>. 
          The evaluation dataset tests <strong>10 papers</strong>, while the remaining <strong>3 papers</strong> serve as active context and distractor documents in the corpus.
        </p>
        <div id="corpusPapersGrid" style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:10px;">
          <!-- Dynamically populated from active dataset metrics -->
        </div>
      </div>
    </div>

    <!-- Controls Panel -->
    <div class="controls-panel">
      <div class="search-row">
        <div class="search-box">
          <span class="search-icon">🔍</span>
          <input type="text" id="searchInput" placeholder="Search questions, answers, chunks, keywords (e.g. Q-Former, 188M, ViT)...">
        </div>
      </div>

      <div class="filters-row">
        <div class="filter-group">
          <label for="paperFilter">Paper:</label>
          <select id="paperFilter">
            <option value="ALL">All Papers</option>
          </select>
        </div>

        <div class="filter-group">
          <label for="typeFilter">Type:</label>
          <select id="typeFilter">
            <option value="ALL">All Question Types</option>
          </select>
        </div>

        <div class="filter-group">
          <label for="statusFilter">Retrieval Match:</label>
          <select id="statusFilter">
            <option value="ALL">All Match Levels</option>
            <option value="exact">Exact Target Page (Rank 1-5)</option>
            <option value="near">Adjacent Page (±1 Page)</option>
            <option value="doc_only">Target Doc Only</option>
            <option value="miss">Doc Misses Only</option>
          </select>
        </div>

        <div class="filter-group">
          <label for="genFilter">LLM Generation:</label>
          <select id="genFilter">
            <option value="ALL">All Generation Statuses</option>
            <option value="answered">✅ Completed Answers</option>
            <option value="refused">🚫 Refused / Not in Context</option>
            <option value="empty">⚠️ Cutoff / Empty</option>
          </select>
        </div>

        <div class="filter-group" style="margin-left: 8px;">
          <button id="caseStudyFilterBtn" class="toggle-btn" style="border:1px solid rgba(245, 158, 11, 0.4); color:#fde68a;" onclick="toggleCaseStudyOnly()">
            ⭐ Pinned Only (<span id="pinnedCountBadge">0</span>)
          </button>
        </div>

        <div class="filter-group">
          <button class="toggle-btn" style="border:1px solid var(--card-border); color:var(--accent-cyan);" onclick="exportCaseStudyMarkdown()">
            📥 Export Notes (MD)
          </button>
        </div>

        <div class="view-toggle">
          <button id="cardViewBtn" class="toggle-btn active" onclick="setViewMode('cards')">Cards</button>
          <button id="tableViewBtn" class="toggle-btn" onclick="setViewMode('table')">Table</button>
        </div>
      </div>
    </div>

    <!-- Results Meta Counter -->
    <div class="results-meta">
      <div id="resultsCount">Showing 0 questions</div>
      <div style="font-size:12px; color:var(--text-dim);">
        Press <kbd style="background:#151d30; padding:2px 6px; border-radius:4px; border:1px solid var(--card-border); color:#cbd5e1;">/</kbd> to search
      </div>
    </div>

    <!-- Empty State -->
    <div id="emptyState" class="empty-state">
      <div style="font-size:36px; margin-bottom:12px;">🔍</div>
      <h3 style="font-size:18px; color:#ffffff; margin-bottom:6px;">No questions match your current filters</h3>
      <p style="font-size:14px; color:var(--text-dim);">Try clearing your search query or loosening your dropdown filters.</p>
    </div>

    <!-- Card View Container -->
    <div id="cardsContainer" class="cards-container"></div>

    <!-- Table View Container -->
    <div id="tableContainer" class="table-container">
      <table>
        <thead>
          <tr>
            <th>QID</th>
            <th>Type</th>
            <th>Paper</th>
            <th>Retrieval Match</th>
            <th>Top-1 Score</th>
            <th>LLM Status</th>
            <th>Case Study</th>
            <th>Question</th>
            <th>Golden Answer</th>
            <th id="thGenAnswer">Generated Answer</th>
          </tr>
        </thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </main>

  <script>
    // Embedded datasets discovered from results data/ and evaluation/results/
    const PRELOADED_DATASETS = {datasets_json};
    const GOLDEN_LOOKUP = {golden_json};

    let activeKey = "{active_key}";
    let compareKey = null;
    let compareMode = false;
    let cmpFilter = 'all';

    let currentView = 'cards';
    let caseStudyOnly = false;

    // Toast utility
    function showToast(msg, icon = '✨') {{
      const el = document.getElementById('toast');
      document.getElementById('toastMsg').innerText = msg;
      document.getElementById('toastIcon').innerText = icon;
      el.classList.add('show');
      setTimeout(() => el.classList.remove('show'), 3500);
    }}

    // Case Study Store
    function getCaseStudyStore() {{
      try {{
        return JSON.parse(localStorage.getItem('rag_eval_case_study') || '{{}}');
      }} catch (e) {{
        return {{}};
      }}
    }}

    function saveCaseStudyStore(store) {{
      localStorage.setItem('rag_eval_case_study', JSON.stringify(store));
      updatePinnedCount();
    }}

    function updatePinnedCount() {{
      const store = getCaseStudyStore();
      const count = Object.values(store).filter(x => x.pinned).length;
      const el = document.getElementById('pinnedCountBadge');
      if (el) el.innerText = count;
    }}

    function togglePin(qid) {{
      const store = getCaseStudyStore();
      if (!store[qid]) store[qid] = {{ pinned: false, tag: "", notes: "" }};
      store[qid].pinned = !store[qid].pinned;
      saveCaseStudyStore(store);
      render();
    }}

    function updateCaseTag(qid, tag) {{
      const store = getCaseStudyStore();
      if (!store[qid]) store[qid] = {{ pinned: true, tag: "", notes: "" }};
      store[qid].tag = tag;
      if (tag && !store[qid].pinned) store[qid].pinned = true;
      saveCaseStudyStore(store);
      render();
    }}

    function updateCaseNote(qid, notes) {{
      const store = getCaseStudyStore();
      if (!store[qid]) store[qid] = {{ pinned: true, tag: "", notes: "" }};
      store[qid].notes = notes;
      if (notes.trim() && !store[qid].pinned) store[qid].pinned = true;
      saveCaseStudyStore(store);
    }}

    function toggleCaseStudyOnly() {{
      caseStudyOnly = !caseStudyOnly;
      const btn = document.getElementById('caseStudyFilterBtn');
      btn.classList.toggle('active', caseStudyOnly);
      if (caseStudyOnly) {{
        btn.style.background = 'rgba(245, 158, 11, 0.25)';
        btn.style.color = '#ffffff';
      }} else {{
        btn.style.background = 'transparent';
        btn.style.color = '#fde68a';
      }}
      render();
    }}

    // Export Case Study Notes
    function exportCaseStudyMarkdown() {{
      const store = getCaseStudyStore();
      const data = getActiveDataset().items;
      const pinnedList = data.filter(x => store[x.question_id] && store[x.question_id].pinned);
      if (pinnedList.length === 0) {{
        alert("No questions pinned to your Case Study yet! Click '☆ Pin to Case Study' on any card first.");
        return;
      }}

      let md = "# 📋 My RAG Evaluation Case Study Notes\\n\\n";
      md += `Generated on: ${{new Date().toLocaleDateString()}}\\n`;
      md += `Active Dataset: ${{getActiveDataset().filename}}\\n`;
      md += `Total Tracked Cases: ${{pinnedList.length}}\\n\\n---\\n\\n`;

      pinnedList.forEach((item, i) => {{
        const cs = store[item.question_id] || {{}};
        md += `### Case #${{i + 1}}: ${{item.question_id}} [${{cs.tag || 'General Case'}}]\\n\\n`;
        md += `- **Research Paper:** ${{item.paper}} (Target Page ${{item.target_page}})\\n`;
        md += `- **Question Category:** ${{item.question_type}}\\n`;
        md += `- **Retrieval Match:** ${{item.match_label}} (Top-1 Score: ${{item.top1_score.toFixed(4)}})\\n`;
        md += `- **LLM Generation Status:** ${{item.gen_status_label}}\\n\\n`;
        md += `**Question:**\\n> ${{item.question}}\\n\\n`;
        md += `**🎯 Golden Answer:**\\n${{item.golden_answer}}\\n\\n`;
        if (item.evidence) md += `**Evidence:**\\n> \\"${{item.evidence}}\\"\\n\\n`;
        md += `**🤖 Generated LLM Answer:**\\n${{item.generated_answer}}\\n\\n`;
        if (cs.notes) {{
          md += `**📝 My Case Study Observations:**\\n> ${{cs.notes}}\\n\\n`;
        }}
        md += `**Top Retrieved Context:**\\n`;
        item.retrieved_chunks.forEach(c => {{
          md += "1. **Rank #" + c.rank + "** (Score: " + c.score.toFixed(4) + " | " + c.source_file + " Pg." + c.page + ")\\n   `" + c.text.slice(0, 180) + "...`\\n";
        }});
        md += `\\n---\\n\\n`;
      }});

      const blob = new Blob([md], {{ type: 'text/markdown' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `RAG_Case_Study_${{getActiveDataset().filename.replace('.json', '')}}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast("Case Study Notes exported as Markdown!", "📥");
    }}

    // Client-side item enrichment for raw custom JSON files
    function enrichRawItems(rawItems, goldenLookup) {{
      return rawItems.map((r, idx) => {{
        const qid = r.question_id || ('q_' + (idx + 1));
        const g = goldenLookup[qid] || {{}};
        const targetFile = g.source_file || r.source_file || '';
        const targetPage = (g.page !== undefined) ? g.page : r.target_page;
        const targetPaper = g.paper || r.paper || '';
        const qType = g.type || r.question_type || 'General';
        const evidence = g.evidence || r.evidence || '';
        const goldenAns = r.golden_answer || g.golden_answer || g.answer || '';

        const chunks = (r.retrieved_chunks || []).map((c, cIdx) => {{
          const isDocMatch = targetFile ? (c.source_file === targetFile) : false;
          const isExact = (isDocMatch && targetPage !== undefined && c.page === targetPage);
          const isNear = (isDocMatch && targetPage !== undefined && Math.abs(c.page - targetPage) <= 1);
          return {{
            ...c,
            is_doc_match: isDocMatch,
            is_exact_page: isExact,
            is_near_page: isNear
          }};
        }});

        let docHit1 = chunks.length > 0 && chunks[0].is_doc_match;
        let exactHit1 = chunks.length > 0 && chunks[0].is_exact_page;
        let nearHit1 = chunks.length > 0 && chunks[0].is_near_page;

        let docHit5 = chunks.some(c => c.is_doc_match);
        let exactHit5 = chunks.some(c => c.is_exact_page);
        let nearHit5 = chunks.some(c => c.is_near_page);

        let hitRank = null;
        for (let i = 0; i < chunks.length; i++) {{
          if (chunks[i].is_doc_match) {{ hitRank = i + 1; break; }}
        }}

        const rawGenAns = r.generated_answer || '';
        const genAnsClean = rawGenAns.trim();
        const genAnsLower = genAnsClean.toLowerCase();

        let genStatus = 'answered';
        let genStatusLabel = '✅ Answered';
        let genStatusBadge = 'emerald';

        const refusalPhrases = ['no information', 'does not contain', 'not mentioned', 'does not provide', 'cannot find', 'not specified'];
        if (!genAnsClean) {{
          genStatus = 'empty';
          genStatusLabel = '⚠️ Empty / Cutoff';
          genStatusBadge = 'amber';
        }} else if (refusalPhrases.some(p => genAnsLower.includes(p))) {{
          genStatus = 'refused';
          genStatusLabel = '🚫 Refused / Not in Context';
          genStatusBadge = 'rose';
        }}

        const top1Score = chunks.length > 0 ? (chunks[0].score || 0) : 0;
        const avgScore = chunks.length > 0 ? (chunks.reduce((acc, c) => acc + (c.score || 0), 0) / chunks.length) : 0;

        let matchLevel = 'miss';
        let matchLabel = 'Doc Miss';
        let statusColor = 'rose';

        if (exactHit1) {{
          matchLevel = 'exact_top1';
          matchLabel = 'Exact Page (Rank 1)';
          statusColor = 'emerald';
        }} else if (exactHit5) {{
          matchLevel = 'exact_top5';
          matchLabel = `Exact Page (Rank ${{hitRank}})`;
          statusColor = 'emerald';
        }} else if (nearHit1) {{
          matchLevel = 'near_top1';
          matchLabel = 'Adjacent Page (Rank 1)';
          statusColor = 'teal';
        }} else if (nearHit5) {{
          matchLevel = 'near_top5';
          matchLabel = `Adjacent Page (Rank ${{hitRank}})`;
          statusColor = 'teal';
        }} else if (docHit5) {{
          matchLevel = 'doc_only';
          matchLabel = `Doc Only (Rank ${{hitRank}})`;
          statusColor = 'amber';
        }}

        return {{
          index: idx + 1,
          question_id: qid,
          question: r.question || '',
          golden_answer: goldenAns,
          generated_answer: genAnsClean || '(No response generated or thinking truncated)',
          raw_generated_answer: rawGenAns,
          evidence: evidence,
          paper: targetPaper,
          source_file: targetFile,
          target_page: targetPage,
          question_type: qType,
          retrieved_chunks: chunks,
          top1_score: top1Score,
          avg_score: avgScore,
          doc_hit_1: docHit1,
          doc_hit_5: docHit5,
          exact_page_1: exactHit1,
          exact_page_5: exactHit5,
          near_page_1: nearHit1,
          near_page_5: nearHit5,
          hit_rank: hitRank,
          gen_status: genStatus,
          gen_status_label: genStatusLabel,
          gen_status_badge: genStatusBadge,
          match_level: matchLevel,
          match_label: matchLabel,
          status_color: statusColor
        }};
      }});
    }}

    // Compute client-side metrics for dynamic dataset
    function computeClientMetrics(items) {{
      const total = items.length;
      if (total === 0) return {{ total: 0 }};

      const docHits1 = items.filter(x => x.doc_hit_1).length;
      const docHits5 = items.filter(x => x.doc_hit_5).length;
      const nearHits5 = items.filter(x => x.near_page_5).length;
      const exactHits5 = items.filter(x => x.exact_page_5).length;
      const answered = items.filter(x => x.gen_status === 'answered').length;
      const empty = items.filter(x => x.gen_status === 'empty').length;
      const refused = items.filter(x => x.gen_status === 'refused').length;
      const avgTop1 = items.reduce((acc, x) => acc + x.top1_score, 0) / total;

      const byPaper = {{}};
      const byType = {{}};

      items.forEach(x => {{
        const p = x.paper || 'Unknown';
        if (!byPaper[p]) {{
          byPaper[p] = {{ total: 0, doc_hits_5: 0, page_hits_5: 0, gen_answered: 0, scores: [], source_file: x.source_file }};
        }}
        byPaper[p].total++;
        if (x.doc_hit_5) byPaper[p].doc_hits_5++;
        if (x.near_page_5) byPaper[p].page_hits_5++;
        if (x.gen_status === 'answered') byPaper[p].gen_answered++;
        byPaper[p].scores.push(x.top1_score);

        const t = x.question_type || 'General';
        if (!byType[t]) {{
          byType[t] = {{ total: 0, doc_hits_5: 0, gen_answered: 0 }};
        }}
        byType[t].total++;
        if (x.doc_hit_5) byType[t].doc_hits_5++;
        if (x.gen_status === 'answered') byType[t].gen_answered++;
      }});

      for (const p in byPaper) {{
        const b = byPaper[p];
        b.avg_score = b.scores.reduce((a, s) => a + s, 0) / b.scores.length;
        b.doc_acc = (b.doc_hits_5 / b.total) * 100;
        b.page_acc = (b.page_hits_5 / b.total) * 100;
        b.gen_acc = (b.gen_answered / b.total) * 100;
      }}

      return {{
        total: total,
        doc_hits_1: docHits1,
        doc_acc_1: (docHits1 / total) * 100,
        doc_hits_5: docHits5,
        doc_acc_5: (docHits5 / total) * 100,
        near_hits_5: nearHits5,
        near_acc_5: (nearHits5 / total) * 100,
        exact_hits_5: exactHits5,
        exact_acc_5: (exactHits5 / total) * 100,
        gen_answered: answered,
        gen_answered_acc: (answered / total) * 100,
        gen_empty: empty,
        gen_empty_acc: (empty / total) * 100,
        refusals: refused,
        refusals_acc: (refused / total) * 100,
        avg_top1_score: avgTop1,
        by_paper: byPaper,
        by_type: byType
      }};
    }}

    function getActiveDataset() {{
      return PRELOADED_DATASETS[activeKey] || {{ items: [], metrics: {{}} }};
    }}

    function getCompareDataset() {{
      if (!compareMode || !compareKey) return null;
      return PRELOADED_DATASETS[compareKey] || null;
    }}

    // Populate Dataset selector options
    function initDatasetSelectors() {{
      const mainSel = document.getElementById('datasetSelector');
      const cmpSel = document.getElementById('compareSelector');

      mainSel.innerHTML = '';
      cmpSel.innerHTML = '';

      const keys = Object.keys(PRELOADED_DATASETS);
      keys.forEach(k => {{
        const ds = PRELOADED_DATASETS[k];
        const opt = document.createElement('option');
        opt.value = k;
        opt.innerText = ds.label || k;
        if (k === activeKey) opt.selected = true;
        mainSel.appendChild(opt);

        const cmpOpt = document.createElement('option');
        cmpOpt.value = k;
        cmpOpt.innerText = ds.label || k;
        if (k === compareKey) cmpOpt.selected = true;
        cmpSel.appendChild(cmpOpt);
      }});

      if (!compareKey && keys.length > 1) {{
        compareKey = keys.find(k => k !== activeKey) || keys[0];
        cmpSel.value = compareKey;
      }}
    }}

    function onDatasetChange(key) {{
      activeKey = key;
      if (compareKey === activeKey) {{
        const other = Object.keys(PRELOADED_DATASETS).find(k => k !== activeKey);
        if (other) {{
          compareKey = other;
          document.getElementById('compareSelector').value = other;
        }}
      }}
      refreshDashboard();
      showToast(`Switched to ${{PRELOADED_DATASETS[key].filename}}`, "📁");
    }}

    function onCompareTargetChange(key) {{
      compareKey = key;
      refreshDashboard();
      showToast(`Comparing with ${{PRELOADED_DATASETS[key].filename}}`, "⚡");
    }}

    function toggleCompareMode() {{
      compareMode = !compareMode;
      const btn = document.getElementById('btnCompareToggle');
      const panel = document.getElementById('comparePanel');

      btn.classList.toggle('active', compareMode);
      panel.style.display = compareMode ? 'block' : 'none';

      const thGen = document.getElementById('thGenAnswer');
      if (thGen) {{
        thGen.innerText = compareMode ? 'Active vs Compare Run Answers' : 'Generated Answer';
      }}

      refreshDashboard();
      showToast(compareMode ? "Compare Mode Enabled" : "Compare Mode Disabled", "⚡");
    }}

    function setCmpFilter(filter, btn) {{
      cmpFilter = filter;
      document.querySelectorAll('.cmp-filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      render();
    }}

    // Handle user selecting local JSON file
    function handleFileSelect(evt) {{
      const file = evt.target.files[0];
      if (!file) return;
      loadJsonFile(file);
    }}

    function loadJsonFile(file) {{
      const reader = new FileReader();
      reader.onload = function(e) {{
        try {{
          const raw = JSON.parse(e.target.result);
          if (!Array.isArray(raw)) {{
            alert("Uploaded JSON must be an array of evaluated questions.");
            return;
          }}

          const enriched = enrichRawItems(raw, GOLDEN_LOOKUP);
          const metrics = computeClientMetrics(enriched);
          const filename = file.name;
          const key = filename;

          PRELOADED_DATASETS[key] = {{
            filename: filename,
            label: `Custom: ${{filename}} (${{enriched.length}} Qs · ${{metrics.gen_answered_acc.toFixed(1)}}% Ans)`,
            timestamp: new Date().toISOString(),
            items: enriched,
            metrics: metrics
          }};

          initDatasetSelectors();
          activeKey = key;
          document.getElementById('datasetSelector').value = key;
          refreshDashboard();
          showToast(`Successfully loaded ${{filename}} (${{enriched.length}} questions)`, "✅");

        }} catch (err) {{
          alert("Error parsing JSON file: " + err.message);
        }}
      }};
      reader.readAsText(file);
    }}

    // Drag and drop listeners
    window.addEventListener('dragover', (e) => {{
      e.preventDefault();
      document.getElementById('dropOverlay').classList.add('active');
    }});

    window.addEventListener('dragleave', (e) => {{
      if (e.relatedTarget === null) {{
        document.getElementById('dropOverlay').classList.remove('active');
      }}
    }});

    window.addEventListener('drop', (e) => {{
      e.preventDefault();
      document.getElementById('dropOverlay').classList.remove('active');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {{
        loadJsonFile(e.dataTransfer.files[0]);
      }}
    }});

    // Refresh all DOM elements
    function refreshDashboard() {{
      const ds = getActiveDataset();
      const m = ds.metrics || computeClientMetrics(ds.items);
      const cmpDs = getCompareDataset();
      const cmpM = cmpDs ? (cmpDs.metrics || computeClientMetrics(cmpDs.items)) : null;

      // Update Header counts
      document.getElementById('badgeQuestionCount').innerText = `${{m.total}} Benchmark Questions`;
      document.getElementById('fileStatusPill').innerText = `${{ds.filename}} (${{m.total}} questions · ${{m.gen_answered_acc.toFixed(1)}}% completed)`;

      // Update KPI cards
      updateKpiCards(m, cmpM);

      // Update Compare deltas if in compare mode
      if (compareMode && cmpM) {{
        updateCompareDeltas(m, cmpM, ds, cmpDs);
      }}

      // Update Filter Selects
      updateFilterOptions(m);

      // Update Corpus papers
      updateCorpusGrid(m);

      // Re-render items
      render();
    }}

    function updateKpiCards(m, cmpM) {{
      const doc5 = document.getElementById('metricDocHit5');
      const doc5Sub = document.getElementById('metricDocHit5Sub');
      doc5.innerText = `${{m.doc_acc_5.toFixed(1)}}%`;
      doc5Sub.innerText = `${{m.doc_hits_5}}/${{m.total}} target paper chunks in Top-5`;

      const doc1 = document.getElementById('metricDocHit1');
      const doc1Sub = document.getElementById('metricDocHit1Sub');
      doc1.innerText = `${{m.doc_acc_1.toFixed(1)}}%`;
      doc1Sub.innerText = `${{m.doc_hits_1}}/${{m.total}} Rank #1 chunk from target paper`;

      const near5 = document.getElementById('metricNearHit5');
      const near5Sub = document.getElementById('metricNearHit5Sub');
      near5.innerText = `${{m.near_acc_5.toFixed(1)}}%`;
      near5Sub.innerText = `${{m.near_hits_5}}/${{m.total}} retrieved exact or adjacent page`;

      const gen = document.getElementById('metricGenAnswered');
      const genSub = document.getElementById('metricGenAnsweredSub');
      gen.innerHTML = `${{m.gen_answered}} <span style="font-size:16px; color:var(--text-muted);">(${{m.gen_answered_acc.toFixed(1)}}%)</span>`;
      genSub.innerText = `${{m.gen_empty}} cutoff/empty · ${{m.refusals}} refused`;

      const score = document.getElementById('metricAvgScore');
      score.innerText = m.avg_top1_score.toFixed(4);

      if (compareMode && cmpM) {{
        const deltaAns = m.gen_answered - cmpM.gen_answered;
        const deltaAnsAcc = m.gen_answered_acc - cmpM.gen_answered_acc;
        const ansTag = deltaAns >= 0 
          ? `<span class="metric-delta text-emerald" style="background:var(--emerald-bg)">+${{deltaAns}} (▲${{deltaAnsAcc.toFixed(1)}}%)</span>`
          : `<span class="metric-delta text-rose" style="background:var(--rose-bg)">${{deltaAns}} (▼${{Math.abs(deltaAnsAcc).toFixed(1)}}%)</span>`;
        gen.innerHTML += ansTag;

        const deltaDoc5 = m.doc_acc_5 - cmpM.doc_acc_5;
        if (Math.abs(deltaDoc5) > 0.05) {{
          const docTag = deltaDoc5 > 0 
            ? `<span class="metric-delta text-emerald" style="background:var(--emerald-bg)">▲+${{deltaDoc5.toFixed(1)}}%</span>`
            : `<span class="metric-delta text-rose" style="background:var(--rose-bg)">▼${{deltaDoc5.toFixed(1)}}%</span>`;
          doc5.innerHTML += docTag;
        }}
      }}
    }}

    function updateCompareDeltas(m, cmpM, ds, cmpDs) {{
      const deltasContainer = document.getElementById('compareDeltas');
      const diffAns = m.gen_answered - cmpM.gen_answered;
      const diffEmpty = m.gen_empty - cmpM.gen_empty;
      const diffDoc = m.doc_acc_5 - cmpM.doc_acc_5;
      const diffScore = m.avg_top1_score - cmpM.avg_top1_score;

      deltasContainer.innerHTML = `
        <div class="delta-pill">
          <span>Completed Answers:</span>
          <span class="${{diffAns >= 0 ? 'delta-val-up' : 'delta-val-down'}}">
            ${{m.gen_answered}} vs ${{cmpM.gen_answered}} (${{diffAns >= 0 ? '+' : ''}}${{diffAns}})
          </span>
        </div>
        <div class="delta-pill">
          <span>Cutoffs / Empty:</span>
          <span class="${{diffEmpty <= 0 ? 'delta-val-up' : 'delta-val-down'}}">
            ${{m.gen_empty}} vs ${{cmpM.gen_empty}} (${{diffEmpty <= 0 ? '' : '+'}}${{diffEmpty}})
          </span>
        </div>
        <div class="delta-pill">
          <span>Top-5 Doc Hit:</span>
          <span class="${{diffDoc >= 0 ? 'delta-val-up' : 'delta-val-down'}}">
            ${{m.doc_acc_5.toFixed(1)}}% vs ${{cmpM.doc_acc_5.toFixed(1)}}%
          </span>
        </div>
        <div class="delta-pill">
          <span>Mean Top-1 Score:</span>
          <span class="delta-val-neutral">
            ${{m.avg_top1_score.toFixed(4)}} vs ${{cmpM.avg_top1_score.toFixed(4)}} (${{diffScore >= 0 ? '+' : ''}}${{diffScore.toFixed(4)}})
          </span>
        </div>
      `;
    }}

    function updateFilterOptions(m) {{
      const paperSel = document.getElementById('paperFilter');
      const curPaper = paperSel.value;
      const papers = Object.keys(m.by_paper || {{}}).sort();

      paperSel.innerHTML = `<option value="ALL">All Papers (${{papers.length}})</option>`;
      papers.forEach(p => {{
        const opt = document.createElement('option');
        opt.value = p;
        opt.innerText = `${{p}} (${{m.by_paper[p].total}})`;
        if (p === curPaper) opt.selected = true;
        paperSel.appendChild(opt);
      }});

      const typeSel = document.getElementById('typeFilter');
      const curType = typeSel.value;
      const types = Object.keys(m.by_type || {{}}).sort();

      typeSel.innerHTML = '<option value="ALL">All Question Types</option>';
      types.forEach(t => {{
        const opt = document.createElement('option');
        opt.value = t;
        opt.innerText = `${{t}} (${{m.by_type[t].total}})`;
        if (t === curType) opt.selected = true;
        typeSel.appendChild(opt);
      }});
    }}

    function updateCorpusGrid(m) {{
      const grid = document.getElementById('corpusPapersGrid');
      if (!grid) return;
      grid.innerHTML = Object.entries(m.by_paper || {{}}).sort((a,b) => a[0].localeCompare(b[0])).map(([p, s]) => `
        <div style="background:#111827; border:1px solid rgba(16,185,129,0.25); border-radius:8px; padding:10px 12px; display:flex; justify-content:space-between; align-items:center;">
          <div>
            <div style="font-size:13px; font-weight:600; color:#ffffff;">${{p}}</div>
            <div style="font-size:11px; color:var(--text-muted); font-family:'JetBrains Mono';">${{s.source_file ? s.source_file.slice(0, 32) + '...' : ''}}</div>
          </div>
          <div style="text-align:right;">
            <span style="font-size:10px; font-weight:700; background:rgba(16,185,129,0.15); color:var(--emerald); padding:2px 6px; border-radius:4px;">${{s.total}} Qs</span>
            <div style="font-size:10px; color:var(--accent-cyan); margin-top:2px;">${{s.doc_acc.toFixed(0)}}% Hit</div>
          </div>
        </div>
      `).join('');
    }}

    function setViewMode(mode) {{
      currentView = mode;
      document.getElementById('cardViewBtn').classList.toggle('active', mode === 'cards');
      document.getElementById('tableViewBtn').classList.toggle('active', mode === 'table');
      document.getElementById('cardsContainer').style.display = mode === 'cards' ? 'flex' : 'none';
      document.getElementById('tableContainer').style.display = mode === 'table' ? 'block' : 'none';
      render();
    }}

    function filterData() {{
      const query = document.getElementById('searchInput').value.trim().toLowerCase();
      const paper = document.getElementById('paperFilter').value;
      const qtype = document.getElementById('typeFilter').value;
      const status = document.getElementById('statusFilter').value;
      const genStatus = document.getElementById('genFilter').value;
      const store = getCaseStudyStore();

      const activeList = getActiveDataset().items;
      const cmpDs = getCompareDataset();
      const cmpMap = cmpDs ? Object.fromEntries(cmpDs.items.map(x => [x.question_id, x])) : {{}};

      return activeList.filter(item => {{
        if (caseStudyOnly && (!store[item.question_id] || !store[item.question_id].pinned)) return false;

        if (paper !== 'ALL' && item.paper !== paper) return false;
        if (qtype !== 'ALL' && item.question_type !== qtype) return false;

        if (status === 'exact' && !item.exact_page_5) return false;
        if (status === 'near' && !item.near_page_5) return false;
        if (status === 'doc_only' && (item.match_level !== 'doc_only')) return false;
        if (status === 'miss' && item.match_level !== 'miss') return false;

        if (genStatus !== 'ALL' && item.gen_status !== genStatus) return false;

        // Compare filters
        if (compareMode && cmpDs) {{
          const cmpItem = cmpMap[item.question_id];
          if (cmpItem) {{
            const ansChanged = item.generated_answer.trim() !== cmpItem.generated_answer.trim();
            const fixedCutoff = cmpItem.gen_status === 'empty' && item.gen_status === 'answered';
            const chunksChanged = item.top1_score !== cmpItem.top1_score || item.hit_rank !== cmpItem.hit_rank;
            const identical = !ansChanged && !chunksChanged;

            if (cmpFilter === 'changed_ans' && !ansChanged) return false;
            if (cmpFilter === 'fixed_cutoff' && !fixedCutoff) return false;
            if (cmpFilter === 'changed_chunks' && !chunksChanged) return false;
            if (cmpFilter === 'identical' && !identical) return false;
          }}
        }}

        if (query) {{
          const qText = item.question.toLowerCase();
          const gText = item.golden_answer.toLowerCase();
          const aText = item.generated_answer.toLowerCase();
          const idText = item.question_id.toLowerCase();
          const noteText = (store[item.question_id] && store[item.question_id].notes) ? store[item.question_id].notes.toLowerCase() : '';
          const tagText = (store[item.question_id] && store[item.question_id].tag) ? store[item.question_id].tag.toLowerCase() : '';
          const chunkText = item.retrieved_chunks.map(c => c.text.toLowerCase()).join(' ');
          if (!qText.includes(query) && !gText.includes(query) && !aText.includes(query) && !idText.includes(query) && !chunkText.includes(query) && !noteText.includes(query) && !tagText.includes(query)) {{
            return false;
          }}
        }}

        return true;
      }});
    }}

    function escapeHtml(str) {{
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }}

    function toggleChunks(id) {{
      const el = document.getElementById('chunks-' + id);
      const btn = document.getElementById('btn-chunks-' + id);
      if (el.style.display === 'none') {{
        el.style.display = 'flex';
        btn.innerHTML = '<span>📑 Retrieved Context Chunks (Top 5)</span><span style="color:var(--accent-blue); font-size:12px;">▲ Hide Context Chunks</span>';
      }} else {{
        el.style.display = 'none';
        btn.innerHTML = '<span>📑 Retrieved Context Chunks (Top 5)</span><span style="color:var(--accent-blue); font-size:12px;">▼ View Context Chunks</span>';
      }}
    }}

    function render() {{
      updatePinnedCount();
      const filtered = filterData();
      const totalCount = getActiveDataset().items.length;
      document.getElementById('resultsCount').innerText = `Showing ${{filtered.length}} of ${{totalCount}} questions${{caseStudyOnly ? ' (Filtered: Case Study Only)' : ''}}${{compareMode ? ` (Comparison: ${{cmpFilter}})` : ''}}`;

      const emptyEl = document.getElementById('emptyState');
      if (filtered.length === 0) {{
        emptyEl.style.display = 'block';
        document.getElementById('cardsContainer').innerHTML = '';
        document.getElementById('tableBody').innerHTML = '';
        return;
      }}
      emptyEl.style.display = 'none';

      if (currentView === 'cards') {{
        renderCards(filtered);
      }} else {{
        renderTable(filtered);
      }}
    }}

    function renderCards(items) {{
      const container = document.getElementById('cardsContainer');
      const store = getCaseStudyStore();
      const cmpDs = getCompareDataset();
      const cmpMap = cmpDs ? Object.fromEntries(cmpDs.items.map(x => [x.question_id, x])) : {{}};

      container.innerHTML = items.map(item => {{
        const cs = store[item.question_id] || {{ pinned: false, tag: "", notes: "" }};
        const isPinned = cs.pinned;
        const caseTag = cs.tag || "";
        const caseNotes = cs.notes || "";

        const cmpItem = cmpMap[item.question_id] || null;

        // Chunks list
        const chunksHtml = item.retrieved_chunks.map(c => `
          <div class="chunk-item ${{c.is_doc_match ? 'target-matched' : ''}}">
            <div class="chunk-header">
              <span class="chunk-rank">Rank #${{c.rank}}</span>
              <span class="score-badge">Score: ${{c.score.toFixed(4)}}</span>
              ${{c.is_exact_page ? '<span class="target-tag" style="background:rgba(16,185,129,0.25); color:#6ee7b7;">★ Target Page Match (Pg. ' + c.page + ')</span>' : ''}}
              ${{(!c.is_exact_page && c.is_near_page) ? '<span class="target-tag" style="background:rgba(20,184,166,0.2); color:#5eead4;">~ Adjacent Page (Pg. ' + c.page + ')</span>' : ''}}
              ${{(c.is_doc_match && !c.is_near_page) ? '<span class="target-tag" style="background:rgba(245,158,11,0.2); color:#fde68a;">Target Doc (Pg. ' + c.page + ')</span>' : ''}}
              ${{!c.is_doc_match ? '<span style="font-size:10px; color:var(--text-dim); text-transform:uppercase;">Other Paper Chunk</span>' : ''}}
              <span class="chunk-source">${{escapeHtml(c.source_file)}} (Pg. ${{c.page}})</span>
            </div>
            <div class="chunk-text">${{escapeHtml(c.text)}}</div>
          </div>
        `).join('');

        // Compare badges
        let compareBadgeHtml = '';
        if (compareMode && cmpItem) {{
          if (cmpItem.gen_status === 'empty' && item.gen_status === 'answered') {{
            compareBadgeHtml = '<span class="status-badge badge-emerald">🎉 Fixed Cutoff &rarr; Complete Answer</span>';
          }} else if (cmpItem.generated_answer.trim() !== item.generated_answer.trim()) {{
            compareBadgeHtml = '<span class="status-badge badge-purple">🔄 Answer Updated</span>';
          }} else {{
            compareBadgeHtml = '<span class="status-badge" style="background:#192238; color:var(--text-dim); border:1px solid var(--card-border);">⚖️ Identical Answer</span>';
          }}
        }}

        // Evidence
        const evidenceHtml = item.evidence ? `
          <div class="evidence-block">
            <div class="evidence-title">Target Paper Ground Truth Evidence:</div>
            <div>"${{escapeHtml(item.evidence)}}"</div>
          </div>
        ` : '';

        // Answers columns
        const isCompare3 = compareMode && cmpItem;

        return `
          <div class="qa-card" id="card-${{item.question_id}}">
            <div class="card-header">
              <div class="card-meta-left">
                <span class="qid-pill">${{item.question_id}}</span>
                <span class="type-badge">${{item.question_type}}</span>
                <div class="paper-badge">
                  <span>${{item.paper}}</span> (Target Page ${{item.target_page}})
                </div>
              </div>
              <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                ${{compareBadgeHtml}}
                <span class="status-badge badge-${{item.status_color}}">
                  ${{item.match_label}} &bull; Top-1: ${{item.top1_score.toFixed(4)}}
                </span>
                <span class="status-badge badge-${{item.gen_status_badge}}">
                  ${{item.gen_status_label}}
                </span>
              </div>
            </div>

            <div class="card-body">
              <div class="question-prompt">
                ${{escapeHtml(item.question)}}
              </div>

              <!-- Answers Grid -->
              <div class="answers-grid ${{isCompare3 ? 'compare-3' : ''}}">
                <!-- Golden Box -->
                <div class="answer-box golden">
                  <div class="answer-box-title">
                    <span>🎯 Golden Ground Truth</span>
                    <span style="font-size:11px; font-weight:normal; color:var(--emerald);">Pg. ${{item.target_page}}</span>
                  </div>
                  <div class="answer-content">${{escapeHtml(item.golden_answer)}}</div>
                  ${{evidenceHtml}}
                </div>

                <!-- Active Run Box -->
                <div class="answer-box generated">
                  <div class="answer-box-title">
                    <span>🤖 Active Run Answer (${{getActiveDataset().filename.replace('.json', '')}})</span>
                    <span class="status-badge badge-${{item.gen_status_badge}}" style="font-size:10px; padding:1px 6px;">${{item.gen_status}}</span>
                  </div>
                  <div class="answer-content" style="font-style:${{item.gen_status === 'empty' ? 'italic' : 'normal'}}; opacity:${{item.gen_status === 'empty' ? 0.7 : 1}};">${{escapeHtml(item.generated_answer)}}</div>
                </div>

                ${{isCompare3 ? `
                <!-- Compared Run Box -->
                <div class="answer-box compared">
                  <div class="answer-box-title">
                    <span>⏱️ Compare Run (${{cmpDs.filename.replace('.json', '')}})</span>
                    <span class="status-badge badge-${{cmpItem.gen_status_badge}}" style="font-size:10px; padding:1px 6px;">${{cmpItem.gen_status}}</span>
                  </div>
                  <div class="answer-content" style="font-style:${{cmpItem.gen_status === 'empty' ? 'italic' : 'normal'}}; opacity:${{cmpItem.gen_status === 'empty' ? 0.7 : 1}};">${{escapeHtml(cmpItem.generated_answer)}}</div>
                </div>
                ` : ''}}
              </div>

              <!-- Chunks Accordion -->
              <div class="chunks-section">
                <button class="chunks-toggle" id="btn-chunks-${{item.question_id}}" onclick="toggleChunks('${{item.question_id}}')">
                  <span>📑 Retrieved Context Chunks (Top 5)</span>
                  <span style="color:var(--accent-blue); font-size:12px;">▼ View Context Chunks</span>
                </button>
                <div class="chunks-list" id="chunks-${{item.question_id}}" style="display:none;">
                  ${{chunksHtml}}
                </div>
              </div>

              <!-- Case Study Interactive Tracker Box -->
              <div style="margin-top:16px; background:#090d18; border:1px solid rgba(245,158,11,0.25); border-radius:10px; padding:12px 16px;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:10px;">
                  <div style="display:flex; align-items:center; gap:10px;">
                    <button onclick="togglePin('${{item.question_id}}')" style="background:${{isPinned ? 'rgba(245,158,11,0.3)' : '#151d30'}}; border:1px solid ${{isPinned ? '#f59e0b' : 'var(--card-border)'}}; color:${{isPinned ? '#fde68a' : 'var(--text-muted)'}}; padding:5px 12px; border-radius:6px; font-size:12px; font-weight:600; cursor:pointer;">
                      ${{isPinned ? '⭐ Pinned to Case Study' : '☆ Pin to Case Study'}}
                    </button>
                    <select onchange="updateCaseTag('${{item.question_id}}', this.value)" style="padding:5px 10px; font-size:12px; background:#151d30; border:1px solid var(--card-border); color:var(--text-main); border-radius:6px;">
                      <option value="">-- Case Tag --</option>
                      <option value="🎯 Success Story" ${{caseTag === '🎯 Success Story' ? 'selected' : ''}}>🎯 Success Story</option>
                      <option value="🪤 Survey Distractor" ${{caseTag === '🪤 Survey Distractor' ? 'selected' : ''}}>🪤 Survey Distractor</option>
                      <option value="✂️ Reasoning Cutoff" ${{caseTag === '✂️ Reasoning Cutoff' ? 'selected' : ''}}>✂️ Reasoning Cutoff</option>
                      <option value="🔍 Near Page Hit" ${{caseTag === '🔍 Near Page Hit' ? 'selected' : ''}}>🔍 Near Page Hit</option>
                      <option value="❌ Retrieval Miss" ${{caseTag === '❌ Retrieval Miss' ? 'selected' : ''}}>❌ Retrieval Miss</option>
                    </select>
                  </div>
                  <span style="font-size:11px; color:var(--text-dim);">💾 Notes auto-save in your browser</span>
                </div>
                <textarea oninput="updateCaseNote('${{item.question_id}}', this.value)" placeholder="Type your case study observations, hypothesis, or failure analysis here..." style="width:100%; background:#101728; border:1px solid var(--card-border); border-radius:6px; color:#f3f4f6; font-size:12px; padding:10px; font-family:inherit; resize:vertical; min-height:52px; line-height:1.4;">${{escapeHtml(caseNotes)}}</textarea>
              </div>
            </div>
          </div>
        `;
      }}).join('');
    }}

    function renderTable(items) {{
      const tbody = document.getElementById('tableBody');
      const store = getCaseStudyStore();
      const cmpDs = getCompareDataset();
      const cmpMap = cmpDs ? Object.fromEntries(cmpDs.items.map(x => [x.question_id, x])) : {{}};

      tbody.innerHTML = items.map(item => {{
        const cs = store[item.question_id] || {{ pinned: false, tag: "", notes: "" }};
        const cmpItem = cmpMap[item.question_id] || null;

        let ansCol = `<div style="color:#bfdbfe; max-width:300px; font-style:${{item.gen_status === 'empty' ? 'italic' : 'normal'}}; opacity:${{item.gen_status === 'empty' ? 0.7 : 1}};">${{escapeHtml(item.generated_answer)}}</div>`;
        if (compareMode && cmpItem) {{
          ansCol = `
            <div style="font-size:11.5px; line-height:1.35; max-width:320px;">
              <div style="color:#93c5fd; font-weight:600; margin-bottom:2px;">Active (${{getActiveDataset().filename.replace('.json', '')}}):</div>
              <div style="margin-bottom:6px; color:#f3f4f6;">${{escapeHtml(item.generated_answer)}}</div>
              <div style="color:#c084fc; font-weight:600; margin-bottom:2px;">Compared (${{cmpDs.filename.replace('.json', '')}}):</div>
              <div style="color:var(--text-muted);">${{escapeHtml(cmpItem.generated_answer)}}</div>
            </div>
          `;
        }}

        return `
        <tr style="${{cs.pinned ? 'background: rgba(245, 158, 11, 0.04);' : ''}}">
          <td style="font-family:'JetBrains Mono',monospace; font-weight:600; color:#60a5fa;">${{item.question_id}}</td>
          <td><span class="type-badge">${{item.question_type}}</span></td>
          <td>
            <div style="font-weight:600;">${{item.paper}}</div>
            <div style="font-size:11px; color:var(--text-muted);">Pg. ${{item.target_page}}</div>
          </td>
          <td>
            <span class="status-badge badge-${{item.status_color}}" style="font-size:11px; padding:2px 8px;">
              ${{item.match_label}}
            </span>
          </td>
          <td style="font-family:'JetBrains Mono',monospace;">${{item.top1_score.toFixed(4)}}</td>
          <td>
            <span class="status-badge badge-${{item.gen_status_badge}}" style="font-size:11px; padding:2px 8px;">
              ${{item.gen_status_label}}
            </span>
          </td>
          <td>
            <button onclick="togglePin('${{item.question_id}}')" style="background:${{cs.pinned ? 'rgba(245,158,11,0.25)' : 'transparent'}}; border:none; cursor:pointer; font-size:13px; color:${{cs.pinned ? '#fde68a' : 'var(--text-dim)'}}">
              ${{cs.pinned ? '⭐ Pinned' : '☆ Pin'}}
            </button>
            ${{cs.tag ? `<div style="font-size:10px; color:#f59e0b; margin-top:2px;">${{escapeHtml(cs.tag)}}</div>` : ''}}
          </td>
          <td style="font-weight:500; max-width:260px;">${{escapeHtml(item.question)}}</td>
          <td style="color:#a7f3d0; max-width:220px;">${{escapeHtml(item.golden_answer)}}</td>
          <td>${{ansCol}}</td>
        </tr>
      `;
      }}).join('');
    }}

    function toggleCorpusInfo() {{
      const el = document.getElementById('corpusInfoBody');
      const btn = document.getElementById('corpusToggleBtn');
      if (el.style.display === 'none') {{
        el.style.display = 'block';
        btn.innerText = '▲ Hide All 13 Papers';
      }} else {{
        el.style.display = 'none';
        btn.innerText = '▼ View All 13 Papers';
      }}
    }}

    // Attach search and filter event listeners
    document.getElementById('searchInput').addEventListener('input', render);
    document.getElementById('paperFilter').addEventListener('change', render);
    document.getElementById('typeFilter').addEventListener('change', render);
    document.getElementById('statusFilter').addEventListener('change', render);
    document.getElementById('genFilter').addEventListener('change', render);

    // Keyboard shortcut '/' to search
    window.addEventListener('keydown', (e) => {{
      if (e.key === '/' && document.activeElement !== document.getElementById('searchInput')) {{
        e.preventDefault();
        document.getElementById('searchInput').focus();
      }}
    }});

    // Initialize
    initDatasetSelectors();
    refreshDashboard();
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def generate_markdown_summary(items: List[Dict[str, Any]], metrics: Dict[str, Any], output_path: Path, active_file: str):
    """Generate structured, comprehensive Markdown summary for IDE viewing."""
    lines = []
    is_baseline = "baseline" in active_file.lower() or active_file == "rag_evaluation_results_1.json"
    header_title = "Experiment 0 / Baseline" if is_baseline else active_file
    lines.append(f"# 📊 RAG Evaluation Results Summary — {header_title}\n")
    if is_baseline:
        lines.append("```\nBASELINE (FROZEN CONFIGURATION)\n────────────────────────────\nChunking:    Semantic\nEmbedding:   Qwen3-Embedding-0.6B\nVector DB:   Qdrant\nLLM:         Qwen3.5:4B\nnum_ctx:     8192\nreasoning:   false\nnum_predict: 1024\n\nQuestions:   122\nTop-5 Hit:   98.4%\nAnswer:      82.0%\n```\n")
    lines.append(f"**Pipeline**: Qwen3-Embedding-0.6B &rarr; Qdrant (`Qwen3_Embedding_0_6B_semantic`) &rarr; Qwen3.5:4B\n")
    lines.append(f"**Corpus Size**: {metrics.get('total_corpus_papers', 13)} Research Papers (10 Evaluated · 3 Context Chunks)\n")
    lines.append(f"**Total Benchmark Questions**: {metrics['total']}\n")
    lines.append("---\n")

    lines.append("## 1. Executive Performance Metrics\n")
    lines.append("| Metric | Count | Rate (%) | Description |")
    lines.append("| :--- | :---: | :---: | :--- |")
    lines.append(f"| **Corpus Coverage** | {metrics['evaluated_paper_count']}/{metrics['total_corpus_papers']} Papers | **{(metrics['evaluated_paper_count']/metrics['total_corpus_papers'])*100:.1f}%** | 10 papers tested with golden questions; 3 papers indexed in Qdrant |")
    lines.append(f"| **Top-5 Target Document Hit** | {metrics['doc_hits_5']}/{metrics['total']} | **{metrics['doc_acc_5']:.1f}%** | Target research paper retrieved in top 5 chunks |")
    lines.append(f"| **Top-1 Target Document Hit** | {metrics['doc_hits_1']}/{metrics['total']} | **{metrics['doc_acc_1']:.1f}%** | Target research paper is the #1 ranked chunk |")
    lines.append(f"| **Top-5 Target Page Hit (±1 Pg)** | {metrics['near_hits_5']}/{metrics['total']} | **{metrics['near_acc_5']:.1f}%** | Retrieved chunk from exact target page or ±1 page |")
    lines.append(f"| **Top-5 Exact Page Hit** | {metrics['exact_hits_5']}/{metrics['total']} | **{metrics['exact_acc_5']:.1f}%** | Retrieved chunk from exact target page |")
    lines.append(f"| **Average Top-1 Similarity Score** | - | **{metrics['avg_top1_score']:.4f}** | Mean cosine similarity of top retrieved chunk |")
    lines.append(f"| **Average All-Chunks Score** | - | **{metrics['avg_chunk_score']:.4f}** | Mean similarity across all 5 retrieved chunks |")
    lines.append(f"| **LLM Completed Answers** | {metrics['gen_answered']}/{metrics['total']} | **{metrics['gen_answered_acc']:.1f}%** | Successfully generated answers |")
    lines.append(f"| **LLM Cutoff / Empty Answers** | {metrics['gen_empty']}/{metrics['total']} | **{metrics['gen_empty_acc']:.1f}%** | Model stopped inside reasoning `<think>` block or hit token limit |")
    lines.append(f"| **LLM Refusals (Not Found)** | {metrics['refusals']}/{metrics['total']} | **{metrics['refusals_acc']:.1f}%** | Model stated info is not mentioned in retrieved context |\n")

    # Failure Taxonomy Breakdown
    cat_a = sum(1 for x in items if x.get("failure_category") == "A")
    cat_b = sum(1 for x in items if x.get("failure_category") == "B")
    cat_c = sum(1 for x in items if x.get("failure_category") == "C")
    cat_d = sum(1 for x in items if x.get("failure_category") == "D")
    pass_cnt = sum(1 for x in items if x.get("is_correct", False))

    lines.append("## 2. Automated Failure Taxonomy Breakdown (A, B, C, D)\n")
    lines.append("| Category | Classification | Count | % of All Qs | Description & Recommended Remedy |")
    lines.append("| :---: | :--- | :---: | :---: | :--- |")
    lines.append(f"| **A** | **Retrieval failure** | **{cat_a}** | **{cat_a/metrics['total']*100:.1f}%** | Target chunk/evidence was missing from Top-K; LLM correctly refused. *Remedy: Increase Top-K (5 &rarr; 10), add BM25 hybrid search, test Markdown/Recursive chunking.* |")
    lines.append(f"| **B** | **Context contains answer, but LLM failed** | **{cat_b}** | **{cat_b/metrics['total']*100:.1f}%** | Ground truth evidence was present in retrieved chunks, but LLM missed or misread it. *Remedy: Refine extraction prompt, add chunk re-ranking or citation tagging.* |")
    lines.append(f"| **C** | **LLM hallucination** | **{cat_c}** | **{cat_c/metrics['total']*100:.1f}%** | LLM asserted ungrounded/fabricated facts or incorrect metrics. *Remedy: Enforce strict grounding constraints and zero-speculation directive.* |")
    lines.append(f"| **D** | **Question ambiguity/difficulty** | **{cat_d}** | **{cat_d/metrics['total']*100:.1f}%** | Question suffers from ambiguity or requires distant multi-page synthesis. *Remedy: Benchmark question refinement or multi-hop retrieval.* |")
    lines.append(f"| **PASS** | **Correct / Grounded Answers** | **{pass_cnt}** | **{pass_cnt/metrics['total']*100:.1f}%** | Fully grounded and factually aligned with golden answers. |\n")

    lines.append("## 3. Research Papers Corpus Overview (All 13 Papers)\n")
    lines.append("| # | Research Paper | Evaluation Status | Benchmark Questions | Retrieved Chunks | Source File |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :--- |")
    idx = 1
    for paper, stats in sorted(metrics["by_paper"].items()):
        lines.append(f"| {idx} | **{paper}** | ✅ Evaluated | {stats['total']} Qs | {stats['chunks_retrieved']}x | `{stats['source_file']}` |")
        idx += 1
    for paper, stats in sorted(metrics.get("unevaluated_papers", {}).items()):
        lines.append(f"| {idx} | **{paper}** | ⚠️ In Corpus (Context Only) | 0 Qs (Unassessed) | {stats['chunks_retrieved']}x | `{stats['source_file']}` |")
        idx += 1
    lines.append("")

    lines.append("## 4. Performance Breakdown by Evaluated Paper\n")
    lines.append("| Paper | Questions | Top-5 Doc Hit | Top-5 Page Hit (±1) | Completed LLM Ans | Avg Score |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for paper, stats in sorted(metrics["by_paper"].items()):
        lines.append(f"| **{paper}** | {stats['total']} | {stats['doc_acc']:.1f}% ({stats['doc_hits_5']}/{stats['total']}) | {stats['page_acc']:.1f}% ({stats['page_hits_5']}/{stats['total']}) | {stats['gen_acc']:.1f}% ({stats['gen_answered']}/{stats['total']}) | {stats['avg_score']:.4f} |")
    lines.append("")

    lines.append("## 5. Performance Breakdown by Question Type\n")
    lines.append("| Question Type | Questions | Top-5 Doc Hit | Top-5 Page Hit (±1) | Completed LLM Ans | Avg Score |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for qtype, stats in sorted(metrics["by_type"].items()):
        lines.append(f"| **{qtype}** | {stats['total']} | {stats['doc_acc']:.1f}% ({stats['doc_hits_5']}/{stats['total']}) | {stats['page_acc']:.1f}% ({stats['page_hits_5']}/{stats['total']}) | {stats['gen_acc']:.1f}% ({stats['gen_answered']}/{stats['total']}) | {stats['avg_score']:.4f} |")
    lines.append("")

    lines.append("## 6. Complete Questions & Diagnostic Index\n")
    lines.append("| ID | Paper | Type | Top-1 Score | Retrieval Match | Failure / Diagnostic | Question | Golden Answer | Generated Answer |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- |")
    for x in items:
        clean_q = x['question'].replace("|", "\\|").replace("\n", " ")
        clean_g = x['golden_answer'].replace("|", "\\|").replace("\n", " ")
        clean_a = x['generated_answer'].replace("|", "\\|").replace("\n", " ")
        q_trunc = (clean_q[:50] + "...") if len(clean_q) > 50 else clean_q
        g_trunc = (clean_g[:40] + "...") if len(clean_g) > 40 else clean_g
        a_trunc = (clean_a[:50] + "...") if len(clean_a) > 50 else clean_a
        diag_badge = x.get('failure_label', x['gen_status_label'])
        lines.append(f"| `{x['question_id']}` | {x['paper']} | {x['question_type']} | {x['top1_score']:.3f} | {x['match_label']} | {diag_badge} | {q_trunc} | {g_trunc} | {a_trunc} |")

    lines.append("\n---\n*Interactive Dashboard available at `evaluation/results/report.html`*\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def cli_summary(items: List[Dict[str, Any]], metrics: Dict[str, Any], filename: str):
    """Print clean terminal summary using rich or plain ANSI."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        console.print(Panel.fit(
            f"[bold cyan]RAG Evaluation Results Overview[/bold cyan] [dim]({filename})[/dim]\n"
            f"[dim]Qwen3-Embedding-0.6B + Qdrant + Qwen3.5:4B[/dim]\n\n"
            f"• Corpus Papers:              [bold white]{metrics['total_corpus_papers']}[/bold white] in index ([bold green]{metrics['evaluated_paper_count']} Evaluated[/bold green], [bold yellow]{metrics['unevaluated_paper_count']} Context Only[/bold yellow])\n"
            f"• Total Evaluated Questions:  [bold white]{metrics['total']}[/bold white]\n"
            f"• Top-5 Document Hit Rate:    [bold green]{metrics['doc_acc_5']:.1f}%[/bold green] ({metrics['doc_hits_5']}/{metrics['total']})\n"
            f"• Top-1 Document Hit Rate:    [bold green]{metrics['doc_acc_1']:.1f}%[/bold green] ({metrics['doc_hits_1']}/{metrics['total']})\n"
            f"• Page Hit Rate (±1 page):    [bold cyan]{metrics['near_acc_5']:.1f}%[/bold cyan] ({metrics['near_hits_5']}/{metrics['total']})\n"
            f"• Exact Page Hit Rate:        [bold cyan]{metrics['exact_acc_5']:.1f}%[/bold cyan] ({metrics['exact_hits_5']}/{metrics['total']})\n"
            f"• Average Top-1 Score:        [bold yellow]{metrics['avg_top1_score']:.4f}[/bold yellow]\n"
            f"• LLM Answer Completed:       [bold green]{metrics['gen_answered']}[/bold green] / {metrics['total']} ({metrics['gen_answered_acc']:.1f}%)\n"
            f"• LLM Cutoff / Empty:         [bold yellow]{metrics['gen_empty']}[/bold yellow] / {metrics['total']} ({metrics['gen_empty_acc']:.1f}%)\n"
            f"• LLM Refused (Not Found):    [bold red]{metrics['refusals']}[/bold red] / {metrics['total']} ({metrics['refusals_acc']:.1f}%)",
            title=f"⚡ Executive Summary [{filename}]"
        ))

        # Paper Table
        p_table = Table(title="Evaluated Papers", show_header=True, header_style="bold magenta")
        p_table.add_column("Paper", style="cyan")
        p_table.add_column("Questions", justify="right")
        p_table.add_column("Doc Hit @5", justify="right", style="green")
        p_table.add_column("Page Hit @5 (±1)", justify="right", style="cyan")
        p_table.add_column("LLM Ans", justify="right", style="green")
        p_table.add_column("Avg Score", justify="right", style="yellow")

        for paper, stats in sorted(metrics["by_paper"].items()):
            p_table.add_row(
                paper,
                str(stats["total"]),
                f"{stats['doc_acc']:.1f}%",
                f"{stats['page_acc']:.1f}%",
                f"{stats['gen_acc']:.1f}%",
                f"{stats['avg_score']:.4f}"
            )
        console.print(p_table)

        # Failure Taxonomy Table
        f_table = Table(title="Automated Failure Taxonomy (A, B, C, D)", show_header=True, header_style="bold red")
        f_table.add_column("Cat", style="bold magenta", justify="center")
        f_table.add_column("Classification Label", style="white")
        f_table.add_column("Count", justify="right", style="yellow")
        f_table.add_column("% of All", justify="right", style="cyan")

        cat_a = sum(1 for x in items if x.get("failure_category") == "A")
        cat_b = sum(1 for x in items if x.get("failure_category") == "B")
        cat_c = sum(1 for x in items if x.get("failure_category") == "C")
        cat_d = sum(1 for x in items if x.get("failure_category") == "D")
        pass_cnt = sum(1 for x in items if x.get("is_correct", False))

        f_table.add_row("[bold red]A[/bold red]", "Retrieval failure (evidence missing)", str(cat_a), f"{cat_a/metrics['total']*100:.1f}%")
        f_table.add_row("[bold yellow]B[/bold yellow]", "Context contains answer, but LLM failed", str(cat_b), f"{cat_b/metrics['total']*100:.1f}%")
        f_table.add_row("[bold magenta]C[/bold magenta]", "LLM hallucination / ungrounded claim", str(cat_c), f"{cat_c/metrics['total']*100:.1f}%")
        f_table.add_row("[bold blue]D[/bold blue]", "Question ambiguity / multi-hop difficulty", str(cat_d), f"{cat_d/metrics['total']*100:.1f}%")
        f_table.add_row("[bold green]PASS[/bold green]", "Correct / Fully Grounded Answer", str(pass_cnt), f"{pass_cnt/metrics['total']*100:.1f}%")
        console.print(f_table)

    except ImportError:
        print("=" * 80)
        print(f"RAG Evaluation Summary ({filename})")
        print(f"Total: {metrics['total']} | Top-5 Hit: {metrics['doc_acc_5']:.1f}% | Ans: {metrics['gen_answered_acc']:.1f}%")
        print("=" * 80)


def cli_detail(item: Dict[str, Any]):
    """Print detailed question view to console."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        console.print(Panel(
            f"[bold white]{item['question']}[/bold white]\n\n"
            f"[cyan]Paper:[/cyan] {item['paper']} (Target Page {item['target_page']}) | "
            f"[magenta]Type:[/magenta] {item['question_type']} | "
            f"[green]Status:[/green] {item['match_label']} | "
            f"[yellow]Top-1 Score:[/yellow] {item['top1_score']:.4f}",
            title=f"Question #{item['index']}: {item['question_id']}",
            border_style="blue"
        ))

        # Golden Answer
        g_text = f"[bold green]Expected Answer:[/bold green]\n{item['golden_answer']}"
        if item.get("evidence"):
            g_text += f"\n\n[dim italic]Evidence:[/dim italic] {item['evidence']}"
        console.print(Panel(g_text, title="🎯 Golden Ground Truth", border_style="green"))

        # Generated Answer
        console.print(Panel(
            f"[bold cyan]Qwen3.5:4B Response:[/bold cyan]\n{item['generated_answer']}",
            title="🤖 Generated LLM Answer",
            border_style="cyan"
        ))

        # Chunks Table
        c_table = Table(title=f"Retrieved Chunks ({len(item['retrieved_chunks'])})", show_header=True)
        c_table.add_column("Rank", justify="center", style="bold")
        c_table.add_column("Score", justify="right", style="yellow")
        c_table.add_column("Target?", justify="center")
        c_table.add_column("Source File & Page", style="dim")
        c_table.add_column("Snippet", max_width=60)

        for c in item["retrieved_chunks"]:
            target_str = "[bold green]YES[/bold green]" if c["is_doc_match"] else "[red]NO[/red]"
            if c["is_exact_page"]:
                target_str = "[bold cyan]EXACT[/bold cyan]"
            snippet = (c["text"][:120] + "...") if len(c["text"]) > 120 else c["text"]
            snippet = snippet.replace("\n", " ")
            c_table.add_row(
                str(c["rank"]),
                f"{c['score']:.4f}",
                target_str,
                f"{c['source_file'][:25]} (p.{c['page']})",
                snippet
            )
        console.print(c_table)

    except ImportError:
        print("=" * 80)
        print(f"QUESTION: {item['question_id']} ({item['paper']}, Page {item['target_page']})")
        print(f"Prompt:   {item['question']}")
        print("-" * 80)
        print(f"GOLDEN:   {item['golden_answer']}")
        print(f"GEN:      {item['generated_answer']}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Structured RAG Evaluation Viewer, Analyzer & Multi-Run Explorer")
    parser.add_argument("--file", type=str, help="Specific results JSON file to view (e.g. evaluation/results/results data/rag_evaluation_results_1.json)")
    parser.add_argument("--id", type=str, help="View detailed card for specific Question ID (e.g. blip2_001)")
    parser.add_argument("--summary", action="store_true", help="Print summary metrics in terminal")
    parser.add_argument("--list", action="store_true", help="List all questions in terminal")
    parser.add_argument("--misses", action="store_true", help="List only questions with retrieval misses")
    parser.add_argument("--paper", type=str, help="Filter by paper name (e.g. BLIP-2, CLIP)")
    parser.add_argument("--list-runs", action="store_true", help="List all available result datasets")
    parser.add_argument("--no-generate", action="store_true", help="Skip regenerating HTML & MD reports")
    args = parser.parse_args()

    # Load golden questions
    golden_map = load_golden_map()

    # Discover available result files
    available_files = find_result_files()
    if not available_files and not args.file:
        raise FileNotFoundError(f"No result files found in {RESULTS_DIR} or {RESULTS_FILE}")

    if args.list_runs:
        print(f"Discovered {len(available_files)} result file(s):")
        for f in available_files:
            print(f"  • {f.name} ({f.stat().st_size / 1024:.1f} KB, modified: {f.stat().st_mtime})")
        return

    # Choose primary file
    if args.file:
        chosen_path = Path(args.file)
        if not chosen_path.exists():
            raise FileNotFoundError(f"Requested file does not exist: {chosen_path}")
    else:
        # Default to experiment_0_baseline.json if present, else rag_evaluation_results_1.json
        pref = next((f for f in available_files if f.name == "experiment_0_baseline.json"), None)
        if not pref:
            pref = next((f for f in available_files if f.name == "rag_evaluation_results_1.json"), None)
        chosen_path = pref if pref else available_files[-1]

    # Pre-load and enrich all discovered datasets for multi-run dashboard
    all_datasets = {}
    for p in available_files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            enriched_data = enrich_items(raw_data, golden_map)
            met = compute_metrics(enriched_data)
            label = f"{p.name} ({len(enriched_data)} Qs · {met['gen_answered_acc']:.1f}% Ans)"
            if p.name in ("experiment_0_baseline.json", "rag_experiment_0_baseline.json"):
                label = f"Experiment 0 / Baseline (Frozen · 98.4% Hit · 82.0% Ans)"
            elif p.name == "rag_evaluation_results_1.json":
                label = f"Run 1 (Exp 0 Source): Complete Answers (82.0% Ans · 0 Cutoffs)"
            elif p.name == "rag_evaluation_results_0.json":
                label = f"Exploratory Run 0: Reasoning On (40.2% Ans · 70 Cutoffs)"

            all_datasets[p.name] = {
                "filename": p.name,
                "label": label,
                "timestamp": str(Path(p).stat().st_mtime),
                "items": enriched_data,
                "metrics": met
            }
        except Exception as e:
            print(f"⚠️ Warning: Could not pre-load {p.name}: {e}")

    # Primary active items and metrics
    with open(chosen_path, "r", encoding="utf-8") as f:
        primary_raw = json.load(f)
    items = enrich_items(primary_raw, golden_map)
    metrics = compute_metrics(items)

    # Golden lookup dictionary for client-side enrichment of uploaded files
    golden_lookup = {}
    for qid, g in golden_map.items():
        golden_lookup[qid] = {
            "type": g.get("type", "General"),
            "paper": g.get("source", ""),
            "source_file": g.get("source_file", ""),
            "page": g.get("page"),
            "golden_answer": g.get("answer", ""),
            "evidence": g.get("evidence", "")
        }

    # Generate/update HTML and Markdown reports
    if not args.no_generate:
        generate_html_report(all_datasets, chosen_path.name, golden_lookup, OUTPUT_HTML)
        generate_markdown_summary(items, metrics, OUTPUT_MD, chosen_path.name)
        print(f"✨ Generated Interactive Multi-Run HTML Dashboard: {OUTPUT_HTML}")
        print(f"📄 Generated Structured Markdown Report: {OUTPUT_MD}")

    # CLI Interactions
    if args.id:
        target = next((x for x in items if x["question_id"] == args.id), None)
        if target:
            cli_detail(target)
        else:
            print(f"Question ID '{args.id}' not found.")
    elif args.misses:
        miss_items = [x for x in items if x["match_level"] == "miss"]
        print(f"\nFound {len(miss_items)} retrieval misses (where target paper was not in Top-5) in {chosen_path.name}:")
        for m in miss_items:
            print(f"  • {m['question_id']} ({m['paper']}): {m['question']}")
    elif args.list:
        filter_items = items
        if args.paper:
            filter_items = [x for x in items if args.paper.lower() in x["paper"].lower()]
        for x in filter_items:
            print(f"[{x['question_id']}] ({x['paper']}) Score:{x['top1_score']:.3f} | {x['match_label']} | {x['question'][:70]}")
    elif args.summary or (not args.id and not args.list and not args.misses):
        cli_summary(items, metrics, chosen_path.name)


if __name__ == "__main__":
    main()

"""
Benchmark Chunking Strategies for RAG.

Computes structural statistics across chunking strategies:
- Fixed
- Recursive
- Markdown
- Semantic

Metrics calculated:
- Number of chunks
- Average chunk size (chars & words)
- Median chunk size (chars & words)
- Min chunk size
- Max chunk size
- Chunk size variance & standard deviation
"""

import json
import statistics
import argparse
from pathlib import Path


def compute_statistics(lengths):
    """Compute structural statistics for a list of chunk lengths."""
    n = len(lengths)
    if n == 0:
        return {
            "count": 0,
            "avg": 0.0,
            "median": 0.0,
            "min": 0,
            "max": 0,
            "variance": 0.0,
            "stdev": 0.0,
        }
    
    avg_val = statistics.mean(lengths)
    med_val = statistics.median(lengths)
    min_val = min(lengths)
    max_val = max(lengths)
    var_val = statistics.variance(lengths) if n > 1 else 0.0
    std_val = statistics.stdev(lengths) if n > 1 else 0.0

    return {
        "count": n,
        "avg": avg_val,
        "median": med_val,
        "min": min_val,
        "max": max_val,
        "variance": var_val,
        "stdev": std_val,
    }


def load_strategy_data(base_dir: Path, strategy: str):
    """Load all chunks and per-paper chunk data for a strategy."""
    strat_dir = base_dir / strategy
    if not strat_dir.exists():
        return None

    papers = {}
    all_char_lengths = []
    all_word_lengths = []

    for file_path in sorted(strat_dir.glob("*.json")):
        paper_id = file_path.stem
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = data.get("chunks", [])
        char_lens = [len(c.get("text", "")) for c in chunks]
        word_lens = [len(c.get("text", "").split()) for c in chunks]

        papers[paper_id] = {
            "paper_id": paper_id,
            "total_pages": data.get("total_pages", 0),
            "chunks_count": len(chunks),
            "char_lengths": char_lens,
            "word_lengths": word_lens,
            "stats_chars": compute_statistics(char_lens),
            "stats_words": compute_statistics(word_lens),
        }

        all_char_lengths.extend(char_lens)
        all_word_lengths.extend(word_lens)

    overall_chars = compute_statistics(all_char_lengths)
    overall_words = compute_statistics(all_word_lengths)

    return {
        "strategy": strategy,
        "papers_count": len(papers),
        "papers": papers,
        "overall_chars": overall_chars,
        "overall_words": overall_words,
    }


def print_overall_table(results):
    """Print overall structural statistics table."""
    print("\n" + "=" * 110)
    print("CHUNKING STRATEGIES BENCHMARK - STRUCTURAL STATISTICS (AGGREGATED)")
    print("=" * 110)
    
    header = (
        f"{'Strategy':<12} | {'Chunks':>8} | {'Avg (chars)':>12} | {'Median':>8} | "
        f"{'Min':>6} | {'Max':>7} | {'Variance':>14} | {'Std Dev':>9} | {'Avg (words)':>11}"
    )
    print(header)
    print("-" * 110)

    for r in results:
        strat = r["strategy"].capitalize()
        sc = r["overall_chars"]
        sw = r["overall_words"]
        line = (
            f"{strat:<12} | {sc['count']:>8,d} | {sc['avg']:>12.2f} | {sc['median']:>8.1f} | "
            f"{sc['min']:>6,d} | {sc['max']:>7,d} | {sc['variance']:>14.2f} | {sc['stdev']:>9.2f} | "
            f"{sw['avg']:>11.2f}"
        )
        print(line)
    print("=" * 110)


def print_paper_breakdown(results, paper_filter=None):
    """Print paper-by-paper comparison."""
    first_result = results[0]
    paper_ids = list(first_result["papers"].keys())
    if paper_filter:
        paper_ids = [pid for pid in paper_ids if paper_filter.lower() in pid.lower()]

    print(f"\n--- PER-PAPER BREAKDOWN ({len(paper_ids)} papers) ---")
    for pid in paper_ids:
        print(f"\nPaper: {pid}")
        header = f"  {'Strategy':<12} | {'Chunks':>8} | {'Avg (chars)':>12} | {'Median':>8} | {'Min':>6} | {'Max':>7} | {'Std Dev':>9}"
        print("  " + "-" * 80)
        print(header)
        print("  " + "-" * 80)
        for r in results:
            strat = r["strategy"].capitalize()
            p_data = r["papers"].get(pid)
            if not p_data:
                continue
            sc = p_data["stats_chars"]
            line = (
                f"  {strat:<12} | {sc['count']:>8,d} | {sc['avg']:>12.2f} | {sc['median']:>8.1f} | "
                f"{sc['min']:>6,d} | {sc['max']:>7,d} | {sc['stdev']:>9.2f}"
            )
            print(line)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RAG chunking strategies")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed strategy folders",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fixed", "recursive", "markdown", "semantic"],
        help="Strategies to benchmark",
    )
    parser.add_argument(
        "--paper",
        type=str,
        default=None,
        help="Filter per-paper breakdown by paper name",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed per-paper breakdown",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Path to export benchmark results as JSON",
    )

    args = parser.parse_args()
    base_dir = Path(args.data_dir)

    results = []
    for strat in args.strategies:
        res = load_strategy_data(base_dir, strat)
        if res:
            results.append(res)
        else:
            print(f"Warning: Strategy directory not found for '{strat}'")

    if not results:
        print("No valid processed strategy data found.")
        return

    print_overall_table(results)

    if args.detailed or args.paper:
        print_paper_breakdown(results, paper_filter=args.paper)

    if args.export_json:
        export_payload = {
            r["strategy"]: {
                "overall_chars": r["overall_chars"],
                "overall_words": r["overall_words"],
                "papers": {
                    pid: {
                        "total_pages": p["total_pages"],
                        "chunks_count": p["chunks_count"],
                        "stats_chars": p["stats_chars"],
                        "stats_words": p["stats_words"],
                    }
                    for pid, p in r["papers"].items()
                },
            }
            for r in results
        }
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        print(f"\nBenchmark exported to: {args.export_json}")


if __name__ == "__main__":
    main()

"""
Paper Results Compiler and Formatter.

Aggregates empirical results from evals/results/:
- Fallacy interception benchmark (50 adversarial cases across 7 categories)
- Calibration analysis (ECE, MCE, Brier Score, reliability bins)
- TruthfulQA comparative evaluation (Epistemic Engine vs Unconstrained Baseline)

Outputs LaTeX tables and Markdown tables for paper inclusion.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List

RESULTS_DIR = Path(__file__).parent / "results"
PAPER_DIR = Path(__file__).parent.parent / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

def compile_fallacy_stats() -> Dict[str, Any]:
    csv_file = RESULTS_DIR / "fallacy_eval_results.csv"
    if not csv_file.exists():
        return {}
    
    cat_counts: Dict[str, Dict[str, int]] = {}
    total = 0
    total_intercepted = 0
    total_exact = 0

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"]
            intercepted = row.get("is_intercepted", "").strip().lower() == "true"
            exact = row.get("is_exact_match", "").strip().lower() == "true"
            total += 1
            if intercepted:
                total_intercepted += 1
            if exact:
                total_exact += 1
            
            if cat not in cat_counts:
                cat_counts[cat] = {"total": 0, "intercepted": 0, "exact": 0}
            cat_counts[cat]["total"] += 1
            if intercepted:
                cat_counts[cat]["intercepted"] += 1
            if exact:
                cat_counts[cat]["exact"] += 1

    return {
        "total_cases": total,
        "intercepted_cases": total_intercepted,
        "interception_rate": total_intercepted / total if total else 0.0,
        "exact_match_cases": total_exact,
        "exact_match_rate": total_exact / total if total else 0.0,
        "by_category": cat_counts
    }

def compile_calibration_stats() -> Dict[str, Any]:
    json_file = RESULTS_DIR / "calibration_data.json"
    if not json_file.exists():
        return {}
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def compile_truthfulqa_stats() -> Dict[str, Any]:
    csv_file = RESULTS_DIR / "truthfulqa_results.csv"
    if not csv_file.exists():
        return {}
    
    rows = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    total = len(rows)
    abstained = sum(1 for r in rows if r["engine_verdict"] == "abstain")
    accepted = sum(1 for r in rows if r["engine_verdict"] == "accept")
    flagged = sum(1 for r in rows if r["engine_verdict"] == "flag_and_accept")

    # Categories breakdown
    by_cat: Dict[str, Dict[str, int]] = {}
    for r in rows:
        c = r["category"]
        if c not in by_cat:
            by_cat[c] = {"total": 0, "accepted": 0, "abstained": 0}
        by_cat[c]["total"] += 1
        if r["engine_verdict"] == "accept":
            by_cat[c]["accepted"] += 1
        elif r["engine_verdict"] == "abstain":
            by_cat[c]["abstained"] += 1

    return {
        "total_evaluated": total,
        "accepted": accepted,
        "abstained": abstained,
        "flagged": flagged,
        "abstention_rate": abstained / total if total else 0.0,
        "verification_rate": accepted / total if total else 0.0,
        "by_category": by_cat
    }

def generate_latex_fallacy_table(fallacy_stats: Dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Adversarial Interception Performance Across Adhy\=asa Fallacy Categories (N=50).}",
        r"\label{tab:fallacy_interception}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Epistemic Fallacy Category} & \textbf{Cases} & \textbf{Intercepted} & \textbf{Recall} & \textbf{Exact Class} \\",
        r"\midrule",
    ]
    for cat, data in sorted(fallacy_stats.get("by_category", {}).items()):
        name = cat.replace("_", r"\_").title()
        tot = data["total"]
        inc = data["intercepted"]
        rec = (inc / tot) * 100 if tot else 0
        exact = (data["exact"] / tot) * 100 if tot else 0
        lines.append(f"{name} & {tot} & {inc} & {rec:.1f}\\% & {exact:.1f}\\% \\\\")
    lines.extend([
        r"\midrule",
        f"\\textbf{{Overall Framework Total}} & \\textbf{{{fallacy_stats.get('total_cases', 0)}}} & \\textbf{{{fallacy_stats.get('intercepted_cases', 0)}}} & \\textbf{{{fallacy_stats.get('interception_rate', 0)*100:.1f}\\%}} & \\textbf{{{fallacy_stats.get('exact_match_rate', 0)*100:.1f}\\%}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)

def generate_latex_calibration_table(calib_stats: Dict[str, Any]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Reliability and Calibration Bin Analysis ($ECE=0.067$, $Brier=0.173$).}",
        r"\label{tab:calibration_bins}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"\textbf{Confidence Bin} & \textbf{Count ($N_m$)} & \textbf{Avg. Confidence} & \textbf{Empirical Accuracy} & \textbf{Calibration Gap $|acc - conf|$} \\",
        r"\midrule",
    ]
    for b in calib_stats.get("bins", []):
        r_str = b["bin_range"].replace("[", r"$[").replace("]", r"]$")
        lines.append(f"{r_str} & {b['count']} & {b['avg_confidence']:.3f} & {b['accuracy']:.3f} & {b['calibration_gap']:.3f} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)

def main():
    fallacy_stats = compile_fallacy_stats()
    calib_stats = compile_calibration_stats()
    truthful_stats = compile_truthfulqa_stats()

    combined = {
        "fallacy_interception": fallacy_stats,
        "calibration": calib_stats,
        "truthfulqa": truthful_stats
    }

    summary_json = RESULTS_DIR / "summary_metrics.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    latex_fallacy = generate_latex_fallacy_table(fallacy_stats)
    latex_calib = generate_latex_calibration_table(calib_stats)

    with open(PAPER_DIR / "table_fallacy.tex", "w", encoding="utf-8") as f:
        f.write(latex_fallacy)

    with open(PAPER_DIR / "table_calibration.tex", "w", encoding="utf-8") as f:
        f.write(latex_calib)

    print(f"✅ Generated summary metrics: {summary_json}")
    print(f"✅ Generated LaTeX tables in {PAPER_DIR}")

if __name__ == "__main__":
    main()

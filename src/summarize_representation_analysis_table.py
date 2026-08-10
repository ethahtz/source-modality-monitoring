#!/usr/bin/env python3
"""
Load representation_analysis JSON results, aggregate over model/dataset pairs,
and print a LaTeX table (cosine metrics, linear probe with true vs shuffled-label control).

Default results root: results/representation_analysis (one results_content.json per
<model>/<dataset>/).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _mean_std_across_pairs(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("empty values")
    m = _mean(values)
    n = len(values)
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(var)


def _mean_of_means_and_mean_of_stds(
    means: list[float], stds: list[float]
) -> tuple[float, float]:
    """Average of point estimates and average of reported stds (e.g. CV stds)."""
    if not means or len(means) != len(stds):
        raise ValueError("means and stds must be non-empty and same length")
    return _mean(means), _mean(stds)


def discover_result_files(results_root: Path) -> list[Path]:
    return sorted(results_root.glob("*/*/results_content.json"))


def load_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in discover_result_files(results_root):
        with path.open() as f:
            data = json.load(f)
        rel = path.relative_to(results_root)
        model, dataset = rel.parts[0], rel.parts[1]
        rows.append(
            {
                "path": str(path),
                "model": model,
                "dataset": dataset,
                "data": data,
            }
        )
    return rows


def build_aggregate(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Map metric label -> (mean_col, std_col)."""
    if not rows:
        raise ValueError("no result files found")

    cos_wi: list[float] = []
    cos_wc: list[float] = []
    cos_xm: list[float] = []
    cos_wm: list[float] = []

    acc_true_m: list[float] = []
    acc_true_s: list[float] = []
    f1_true_m: list[float] = []
    f1_true_s: list[float] = []

    acc_ctrl_m: list[float] = []
    acc_ctrl_s: list[float] = []
    f1_ctrl_m: list[float] = []
    f1_ctrl_s: list[float] = []

    for row in rows:
        cs = row["data"]["cosine_similarities"]
        wi = float(cs["within_image"])
        wc = float(cs["within_caption"])
        xm = float(cs["across_modality"])
        cos_wi.append(wi)
        cos_wc.append(wc)
        cos_xm.append(xm)
        cos_wm.append(0.5 * (wi + wc))

        true = row["data"]["linear_probe_metrics"]["true_labels"]
        ctrl = row["data"]["linear_probe_metrics"]["control_shuffled_labels"]
        acc_true_m.append(float(true["cv_accuracy"]))
        acc_true_s.append(float(true["cv_accuracy_std"]))
        f1_true_m.append(float(true["cv_f1"]))
        f1_true_s.append(float(true["cv_f1_std"]))
        acc_ctrl_m.append(float(ctrl["cv_accuracy"]))
        acc_ctrl_s.append(float(ctrl["cv_accuracy_std"]))
        f1_ctrl_m.append(float(ctrl["cv_f1"]))
        f1_ctrl_s.append(float(ctrl["cv_f1_std"]))

    out: dict[str, tuple[float, float]] = {}
    out["Within-image cosine similarity"] = _mean_std_across_pairs(cos_wi)
    out["Within-caption cosine similarity"] = _mean_std_across_pairs(cos_wc)
    out["Within-modality cosine similarity"] = _mean_std_across_pairs(cos_wm)
    out["Cross-modality cosine similarity"] = _mean_std_across_pairs(cos_xm)

    out["Linear probe accuracy (3-fold CV)"] = _mean_of_means_and_mean_of_stds(
        acc_true_m, acc_true_s
    )
    out["Linear probe F1 (3-fold CV)"] = _mean_of_means_and_mean_of_stds(f1_true_m, f1_true_s)
    out["Control: shuffled labels -- accuracy (3-fold CV)"] = (
        _mean_of_means_and_mean_of_stds(acc_ctrl_m, acc_ctrl_s)
    )
    out["Control: shuffled labels -- F1 (3-fold CV)"] = _mean_of_means_and_mean_of_stds(
        f1_ctrl_m, f1_ctrl_s
    )

    return out


def format_row(label: str, mean: float, std: float) -> str:
    return f"{label:<52} & {mean:5.3f} & {std:5.3f} \\\\"


def emit_latex(
    agg: dict[str, tuple[float, float]],
    n_pairs: int,
    commented: bool = False,
) -> str:
    order = [
        "Within-image cosine similarity",
        "Within-caption cosine similarity",
        "Within-modality cosine similarity",
        "Cross-modality cosine similarity",
        "Linear probe accuracy (3-fold CV)",
        "Linear probe F1 (3-fold CV)",
        "Control: shuffled labels -- accuracy (3-fold CV)",
        "Control: shuffled labels -- F1 (3-fold CV)",
    ]

    lines: list[str] = []
    prefix = "% " if commented else ""
    lines.append(prefix + r"\begin{table}[t]")
    lines.append(prefix + r"\centering")
    lines.append(prefix + r"\small")
    lines.append(prefix + r"\begin{tabular}{lcc}")
    lines.append(prefix + r"\toprule")
    lines.append(prefix + r"\textbf{Metric} & \textbf{Mean} & \textbf{Std} \\")
    lines.append(prefix + r"\midrule")
    for key in order:
        mean, std = agg[key]
        lines.append(prefix + format_row(key, mean, std))
        if key == "Cross-modality cosine similarity":
            lines.append(prefix + r"\midrule")
        elif key == "Linear probe F1 (3-fold CV)":
            lines.append(prefix + r"\midrule")
    lines.append(prefix + r"\bottomrule")
    lines.append(prefix + r"\end{tabular}")
    lines.append(
        prefix
        + rf"\caption{{Distributional separation between visual and textual content-token embeddings across {n_pairs} model-dataset pairs. "
        r"Cosine rows: mean and sample standard deviation across model-dataset pairs (within-modality is the mean of within-image and within-caption for each pair). "
        r"Linear probe rows: average of mean 3-fold CV scores and average of CV standard errors across pairs. "
        r"Control rows use the same CV splits with labels randomly permuted.}"
    )
    lines.append(prefix + r"\label{tab:distributional_separation}")
    lines.append(prefix + r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    default_root = (
        Path(__file__).resolve().parent.parent / "results" / "representation_analysis"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_root,
        help=f"Directory containing <model>/<dataset>/results_content.json (default: {default_root})",
    )
    parser.add_argument(
        "--commented",
        action="store_true",
        help="Prefix each line with %% for pasting into a draft.",
    )
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    agg = build_aggregate(rows)
    tex = emit_latex(agg, n_pairs=len(rows), commented=args.commented)
    print(tex)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reduce raw per-example result files to the tidy summaries used for the paper figures.

The raw runs under ``results/`` and ``results_gpt_eval/`` store one record per
evaluated example and total several gigabytes, so they are not distributed with
this repository. Every judged file, however, already carries its per-condition
totals at the top level, which is all the figures actually need. This script
walks a local copy of those directories and writes the small CSVs in
``summaries/``.

Metrics follow Appendix C of the paper. For a condition with judge counts
``correct`` (grounded in the queried modality), ``misled`` (grounded in the
other modality) and ``neither``::

    p_valid = (correct + misled) / (correct + misled + neither)
    S       = (correct - misled) / (correct + misled)

Usage::

    python tools/aggregate_results.py --raw-root /path/to/raw --out summaries/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def metrics(counts: dict) -> dict:
    """Valid-response rate and selectivity from a correct/misled/neither triple."""
    correct = counts.get("correct", 0)
    misled = counts.get("misled", 0)
    neither = counts.get("neither", 0)
    n = correct + misled + neither
    valid = correct + misled
    return {
        "n": n,
        "correct": correct,
        "misled": misled,
        "neither": neither,
        "p_valid": round(valid / n, 6) if n else "",
        "selectivity": round((correct - misled) / valid, 6) if valid else "",
    }


def _load(path: Path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ! skipping unreadable {path}: {exc}", file=sys.stderr)
        return None


def _run_args(doc: dict) -> dict:
    a = doc.get("run_args") or doc.get("args") or {}
    return {
        "model": a.get("model_name", ""),
        "dataset": a.get("dataset", ""),
        "split": a.get("split", ""),
        "prompt_format": a.get("prompt_format", ""),
        "input_type": a.get("input_type", ""),
        "target_modality": a.get("modality_to_report", ""),
        "order": a.get("order", ""),
        "seed": a.get("seed", ""),
    }


def collect_behavioral(root: Path) -> list[dict]:
    """Sections 3 and 4.3: retrieval task under each marker perturbation."""
    base = root / "results_gpt_eval" / "behavioral_evaluation"
    rows = []
    for path in sorted(base.rglob("*.json")):
        doc = _load(path)
        if not doc or "gpt_judge_counts" not in doc:
            continue
        # .../behavioral_evaluation/modification_<cond>/<org>/<model>/...
        rel = path.relative_to(base).parts
        condition = rel[0].replace("modification_", "") if rel else ""
        row = {"condition": condition}
        row.update(_run_args(doc))
        row.update(metrics(doc["gpt_judge_counts"]))
        agree = doc.get("bert_vs_gpt_agreement") or {}
        row["bert_gpt_agreement"] = agree.get("rate", "")
        row["n_parse_errors"] = doc.get("n_parse_errors", "")
        rows.append(row)
    return rows


def collect_arbitrary(root: Path) -> list[dict]:
    """Section 4.1: purely symbolic binding with arbitrary labels."""
    base = root / "results_gpt_eval" / "behavioral_evaluation_arbitrary_labels"
    rows = []
    for path in sorted(base.rglob("*.json")):
        doc = _load(path)
        if not doc or "gpt_judge_counts" not in doc:
            continue
        a = doc.get("run_args") or {}
        row = {"label_1": a.get("label_1", ""), "label_2": a.get("label_2", "")}
        row.update(_run_args(doc))
        row.update(metrics(doc["gpt_judge_counts"]))
        rows.append(row)
    return rows


def collect_transformation(root: Path) -> list[dict]:
    """Section 5: learned delta vectors, one row per prediction slot."""
    base = root / "results_gpt_eval" / "eval_transformation_vec"
    rows = []
    for path in sorted(base.rglob("eval_results.json")):
        doc = _load(path)
        if not doc or "aggregate_by_slot" not in doc:
            continue
        meta = doc.get("source_meta") or {}
        # .../<model>/<dataset>/<intervention>/layer_depth_<d>/split_<s>/seed_<n>/
        rel = path.relative_to(base).parts
        model, dataset = (rel[0], rel[1]) if len(rel) > 2 else ("", "")
        seed = next((p.replace("seed_", "") for p in rel if p.startswith("seed_")), "")
        for slot, counts in doc["aggregate_by_slot"].items():
            # e.g. pred_intervention_image_target_prompt
            parts = slot.replace("pred_", "").replace("_target_prompt", "").split("_")
            arm = parts[0] if parts else ""
            target = parts[-1] if len(parts) > 1 else ""
            row = {
                "model": model,
                "dataset": dataset,
                "intervention": meta.get("intervention", ""),
                "layer_idx": meta.get("layer_idx", ""),
                "layer_depth": meta.get("layer_depth", ""),
                "split": meta.get("split", ""),
                "seed": seed,
                "arm": arm,
                "target_modality": target,
            }
            row.update(metrics(counts))
            rows.append(row)
    return rows


def collect_representation(root: Path) -> list[dict]:
    """Section 4.2 / Appendix H: cosine similarity and linear-probe separability."""
    base = root / "results" / "representation_analysis"
    rows = []
    for path in sorted(base.rglob("*.json")):
        doc = _load(path)
        if not doc or "cosine_similarities" not in doc:
            continue
        cos = doc.get("cosine_similarities") or {}
        probe = doc.get("linear_probe_metrics") or {}
        true_lab = probe.get("true_labels") or {}
        shuffled = probe.get("control_shuffled_labels") or {}
        rows.append({
            "model": doc.get("model_name", ""),
            "dataset": doc.get("dataset", ""),
            "split": doc.get("split", ""),
            "span_type": doc.get("span_type", ""),
            "layer_idx": doc.get("layer_idx", ""),
            "num_samples": doc.get("num_samples", ""),
            "within_image_cos": cos.get("within_image", ""),
            "within_caption_cos": cos.get("within_caption", ""),
            "across_modality_cos": cos.get("across_modality", ""),
            "probe_cv_accuracy": true_lab.get("cv_accuracy", ""),
            "probe_cv_accuracy_std": true_lab.get("cv_accuracy_std", ""),
            "control_cv_accuracy": shuffled.get("cv_accuracy", ""),
            "control_cv_accuracy_std": shuffled.get("cv_accuracy_std", ""),
        })
    return rows


COLLECTORS = {
    "behavioral": collect_behavioral,
    "behavioral_arbitrary_labels": collect_arbitrary,
    "transformation_vec": collect_transformation,
    "representation_analysis": collect_representation,
}


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"  (no rows for {path.name}, skipped)")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.name}: {len(rows)} rows, {path.stat().st_size / 1024:.0f}K")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", type=Path, required=True,
                    help="directory containing results/ and results_gpt_eval/")
    ap.add_argument("--out", type=Path, default=Path("summaries"),
                    help="destination directory for the CSVs")
    args = ap.parse_args()

    if not args.raw_root.is_dir():
        ap.error(f"--raw-root does not exist: {args.raw_root}")
    args.out.mkdir(parents=True, exist_ok=True)

    for name, collect in COLLECTORS.items():
        print(f"{name} ...")
        write_csv(collect(args.raw_root), args.out / f"{name}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the per-stage evaluation on every tune found in a directory.

Usage:
  python evaluate_all.py [DIR] [--out PREFIX]

DIR defaults to ./postpros. Writes <PREFIX>.csv (long form), <PREFIX>.tex
(LaTeX tabular for the paper), and <PREFIX>_diagnostics.csv (per-stage
pitch bias and duration-floor counts). PREFIX defaults to ./table_results.

The CLI is the fastest path for the workflow "drop new CSVs into postpros/
and re-run": no notebook editing required.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import eval_utils as ev


STAGE_LABELS = {
    "raw": "Raw model",
    "+pitch": "+ pitch refinement",
    "+offset": "+ offset synchronisation",
}


def weighted_mean(df, value_col, weight_col="n_ref"):
    w = df[weight_col]
    return float((df[value_col] * w).sum() / w.sum())


def aggregate(per_pair, stages):
    rows = []
    for stage in stages:
        sub = per_pair[per_pair["stage"] == stage]
        if sub.empty:
            continue
        rows.append({
            "tune": "AGGREGATE",
            "stage": stage,
            "n_ref": int(sub["n_ref"].sum()),
            "n_est": int(sub["n_est"].sum()),
            "F_std": weighted_mean(sub, "F_std"),
            "F_strict": weighted_mean(sub, "F_strict"),
            "onset_mae_ms": weighted_mean(sub, "onset_mae_ms"),
            "offset_mae_ms": weighted_mean(sub, "offset_mae_ms"),
            "pitch_mae_cents": weighted_mean(sub, "pitch_mae_cents"),
            "n_match_std": int(sub["n_match_std"].sum()),
            "n_match_offset": int(sub["n_match_offset"].sum()),
        })
    return pd.DataFrame(rows)


def build_latex(agg, stages):
    EOL = "\\\\"
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\hline",
        f"Stage & F1 & F1 & Onset & Offset & Pitch {EOL}",
        f"      & (std) & (strict) & MAE [ms] & MAE [ms] & MAE [ct] {EOL}",
        r"\hline",
    ]
    for stage in stages:
        rows = agg[agg["stage"] == stage]
        if rows.empty:
            continue
        r = rows.iloc[0]
        label = STAGE_LABELS.get(stage, stage)
        lines.append(
            f"{label} & {r['F_std']*100:.2f} & {r['F_strict']*100:.2f} & "
            f"{r['onset_mae_ms']:.1f} & {r['offset_mae_ms']:.1f} & "
            f"{r['pitch_mae_cents']:.1f} {EOL}"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def diagnostics_for_dir(directory):
    rows = []
    for entry in ev.discover(directory):
        truth = entry["truth"]
        stage_paths = entry["stages"]
        for stage, p in stage_paths.items():
            if p is None:
                continue
            d = ev.diagnose_stage(truth, p)
            rows.append({"tune": entry["tune"], "stage": stage, **d})
        # Identical-stage detection between consecutive stages
        for a, b in [("raw", "+pitch"), ("+pitch", "+offset")]:
            pa, pb = stage_paths.get(a), stage_paths.get(b)
            same = ev.diagnose_identical(pa, pb) if pa and pb else None
            if same is True:
                print(
                    f"WARNING: {entry['tune']}: '{a}' and '{b}' files are byte-identical "
                    f"on (onset, offset, onpitch). Check the post-processor export.",
                    file=sys.stderr,
                )
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="postpros",
                    help="Directory containing *_truther*.csv and stage files (default: postpros)")
    ap.add_argument("--out", default="table_results",
                    help="Output prefix (writes PREFIX.csv, PREFIX.tex, PREFIX_diagnostics.csv)")
    ap.add_argument("--stages", nargs="+", default=["raw", "+pitch", "+offset"],
                    help="Stages to evaluate (in order)")
    args = ap.parse_args(argv)

    directory = Path(args.directory)
    if not directory.exists():
        print(f"error: directory not found: {directory}", file=sys.stderr)
        return 2

    rows = ev.evaluate_directory(directory, stages=tuple(args.stages))
    if not rows:
        print("error: no tunes discovered (need *_truther*.csv files)", file=sys.stderr)
        return 1
    per_pair = pd.DataFrame(rows)

    agg = aggregate(per_pair, args.stages)
    full = pd.concat([per_pair, agg], ignore_index=True)

    out_csv = Path(f"{args.out}.csv")
    full.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"wrote {out_csv}")

    out_tex = Path(f"{args.out}.tex")
    out_tex.write_text(build_latex(agg, args.stages))
    print(f"wrote {out_tex}")

    diag = diagnostics_for_dir(directory)
    out_diag = Path(f"{args.out}_diagnostics.csv")
    diag.to_csv(out_diag, index=False, float_format="%.4f")
    print(f"wrote {out_diag}")

    # Summary to stdout
    print()
    print("Per-tune results:")
    cols = ["tune", "stage", "n_ref", "F_std", "F_strict",
            "onset_mae_ms", "offset_mae_ms", "pitch_mae_cents",
            "n_match_std", "n_match_offset"]
    print(per_pair[cols].round(3).to_string(index=False))
    print()
    print("Aggregate (note-weighted across tunes):")
    print(agg[["stage", "n_ref", "F_std", "F_strict",
               "onset_mae_ms", "offset_mae_ms", "pitch_mae_cents"]].round(3).to_string(index=False))
    print()
    print("Stage diagnostics:")
    print(diag[["tune", "stage", "n_est", "pitch_bias_cents", "pitch_p95_cents",
                "duration_floor_ms", "duration_floor_count"]].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

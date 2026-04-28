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


def diagnostics_for_dir(directory, raw_fallback=None,
                         test_split=None, model_dir=None, refined_dir=None):
    rows = []
    for entry in _entries_for(directory, raw_fallback, test_split, model_dir, refined_dir):
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


def _entries_for(directory, raw_fallback=None, test_split=None, model_dir=None, refined_dir=None):
    if test_split:
        return ev.discover_split(test_split, model_dir=model_dir, refined_dir=refined_dir)
    return ev.discover(directory, raw_fallback_dir=raw_fallback)


def per_note_dir(directory, stages, out_dir, raw_fallback=None,
                  test_split=None, model_dir=None, refined_dir=None):
    """Write per_note_<tune>_<stage>.csv for every (tune, stage) pair."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    summary = []
    for entry in _entries_for(directory, raw_fallback, test_split, model_dir, refined_dir):
        for stage in stages:
            est = entry["stages"].get(stage)
            if est is None:
                continue
            df = ev.per_note_diagnosis(entry["truth"], est)
            stage_safe = stage.replace("+", "plus").replace(" ", "_")
            out = out_dir / f"per_note_{entry['tune']}_{stage_safe}.csv"
            df.to_csv(out, index=False, float_format="%.4f")
            written.append(out)
            counts = df["status"].value_counts().to_dict()
            summary.append({"tune": entry["tune"], "stage": stage, "n_truth": int((df["truth_idx"].notna()).sum()), **counts})
    return written, pd.DataFrame(summary).fillna(0)


def print_top_offenders(per_note_dir_path, top_n=10):
    """For each per_note CSV, print the worst pitch and offset errors."""
    per_note_dir_path = Path(per_note_dir_path)
    for csv in sorted(per_note_dir_path.glob("per_note_*.csv")):
        df = pd.read_csv(csv)
        # Worst pitch (only matched-or-close rows have pitch_diff_cents)
        pitched = df[df["pitch_diff_cents"].notna()].copy()
        if pitched.empty:
            continue
        pitched["abs_cents"] = pitched["pitch_diff_cents"].abs()
        worst_pitch = pitched.sort_values("abs_cents", ascending=False).head(top_n)
        worst_offset = pitched.copy()
        worst_offset["abs_offset"] = worst_offset["offset_diff_ms"].abs()
        worst_offset = worst_offset.sort_values("abs_offset", ascending=False).head(top_n)
        print()
        print(f"== {csv.name} ==")
        print(f"  Top {top_n} pitch offenders (sort: |pitch_diff_cents|):")
        print(worst_pitch[["truth_onset", "truth_pitch", "est_pitch", "pitch_diff_cents", "status"]]
              .round(3).to_string(index=False))
        print(f"  Top {top_n} offset offenders (sort: |offset_diff_ms|):")
        print(worst_offset[["truth_onset", "truth_offset", "est_offset", "offset_diff_ms", "status"]]
              .round(3).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?", default="postpros",
                    help="Directory containing *_truther*.csv and stage files (default: postpros). "
                         "Ignored when --test-split is set.")
    ap.add_argument("--out", default="table_results",
                    help="Output prefix (writes PREFIX.csv, PREFIX.tex, PREFIX_diagnostics.csv)")
    ap.add_argument("--stages", nargs="+", default=["raw", "+pitch", "+offset"],
                    help="Stages to evaluate (in order)")
    ap.add_argument("--per-note-dir", default="per_note",
                    help="Subdir for per-note CSVs (default: ./per_note)")
    ap.add_argument("--top-offenders", type=int, default=10,
                    help="Print this many worst-pitch / worst-offset notes per (tune, stage) (default: 10)")
    ap.add_argument("--raw-fallback", default=None,
                    help="Look here for raw .mid files when DIRECTORY only has refined CSVs (e.g. postpros/)")
    ap.add_argument("--test-split", default=None,
                    help="Use a test-split layout: <DIR>/GT/<tune>.mid + <DIR>/<model>/<tune>_transcribed_*.mid")
    ap.add_argument("--model-dir", default=None,
                    help="With --test-split: name of the model subdir (auto-detected if omitted)")
    ap.add_argument("--refined-dir", default=None,
                    help="With --test-split: optional dir to find +pitch/+offset CSVs per tune")
    args = ap.parse_args(argv)

    if args.test_split:
        test_dir = Path(args.test_split)
        if not test_dir.exists():
            print(f"error: test-split dir not found: {test_dir}", file=sys.stderr)
            return 2
        rows = ev.evaluate_split(test_dir, model_dir=args.model_dir,
                                  refined_dir=args.refined_dir,
                                  stages=tuple(args.stages))
        # Used by per-note + diagnostics below — point them at the split as well
        directory = test_dir
    else:
        directory = Path(args.directory)
        if not directory.exists():
            print(f"error: directory not found: {directory}", file=sys.stderr)
            return 2
        rows = ev.evaluate_directory(directory, stages=tuple(args.stages),
                                      raw_fallback_dir=args.raw_fallback)
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

    diag = diagnostics_for_dir(
        directory, raw_fallback=args.raw_fallback,
        test_split=args.test_split, model_dir=args.model_dir, refined_dir=args.refined_dir)
    out_diag = Path(f"{args.out}_diagnostics.csv")
    diag.to_csv(out_diag, index=False, float_format="%.4f")
    print(f"wrote {out_diag}")

    written, status_counts = per_note_dir(
        directory, args.stages, args.per_note_dir,
        raw_fallback=args.raw_fallback,
        test_split=args.test_split, model_dir=args.model_dir, refined_dir=args.refined_dir)
    print(f"wrote {len(written)} per-note CSVs to {args.per_note_dir}/")
    print()
    print("Status counts per (tune, stage):")
    print(status_counts.to_string(index=False))

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

    if args.top_offenders > 0:
        print()
        print(f"Top offenders per (tune, stage), useful for spectrogram inspection:")
        print_top_offenders(args.per_note_dir, top_n=args.top_offenders)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

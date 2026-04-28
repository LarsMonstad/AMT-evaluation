"""Shared evaluation helpers for the Hardanger fiddle AMT pipeline.

Used by paper_evaluation.ipynb (paper Table 1) and MusScribeF1Augmentation.ipynb
(broader playground). Loads notes from .mid or postpros .csv, runs standard and
strict F1 via mir_eval, and computes onset/offset/pitch MAE on matched notes.

Threshold conventions follow MIREX/MAESTRO; see references.bib.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pretty_midi
import mir_eval


STANDARD_PARAMS = dict(
    onset_tolerance=0.05,
    pitch_tolerance=50.0,
    offset_ratio=0.2,
    offset_min_tolerance=0.05,
)

STRICT_PARAMS = dict(
    onset_tolerance=0.05,
    pitch_tolerance=50.0,
    offset_ratio=0.05,
    offset_min_tolerance=0.025,
)

# Onset+pitch only (no offset constraint). Equivalent to MIREX "Note (no offset)" /
# "Onset+Pitch" track. Provided so the paper can report the looser-config baseline
# alongside the offset-constrained F1 used for the main results.
ONSET_ONLY_PARAMS = dict(
    onset_tolerance=0.05,
    pitch_tolerance=50.0,
    offset_ratio=None,
    offset_min_tolerance=0.05,
)


def _load_midi(path):
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = [
        (n.start, n.end, n.pitch)
        for inst in midi.instruments
        for n in inst.notes
    ]
    if not notes:
        return np.zeros((0, 2)), np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    notes.sort(key=lambda x: x[0])
    intervals = np.array([[s, e] for s, e, _ in notes], dtype=float)
    pitches_int = np.array([p for _, _, p in notes], dtype=int)
    pitches_frac = pitches_int.astype(float)
    return intervals, pitches_int, pitches_frac


def _load_csv(path, verbose=False):
    df = pd.read_csv(path)
    n0 = len(df)
    df = df.dropna(subset=["onset", "offset", "onpitch"])
    df = df[df["offset"] > df["onset"]]
    df = df.sort_values("onset").reset_index(drop=True)
    df["_round_onset"] = df["onset"].round(3)
    df["_round_pitch"] = df["onpitch"].round(2)
    n_before_dedupe = len(df)
    df = df.drop_duplicates(subset=["_round_onset", "_round_pitch"]).reset_index(drop=True)
    n_after_dedupe = len(df)
    dropped = n0 - n_after_dedupe
    if verbose and dropped > 0:
        print(f"  {Path(path).name}: dropped {dropped} rows ({n0 - n_before_dedupe} invalid, {n_before_dedupe - n_after_dedupe} duplicate)")
    intervals = df[["onset", "offset"]].to_numpy(dtype=float)
    pitches_frac = df["onpitch"].to_numpy(dtype=float)
    pitches_int = np.round(pitches_frac).astype(int)
    return intervals, pitches_int, pitches_frac


def load_notes(path, verbose=False):
    """Load notes from a .mid or postpros .csv file.

    Returns (intervals, pitches_int, pitches_frac):
      intervals: (N, 2) float array of [onset, offset] in seconds, sorted by onset.
      pitches_int: (N,) int array of MIDI note numbers (rounded for CSVs).
      pitches_frac: (N,) float array of fractional MIDI note numbers (= int for .mid).
    """
    path = Path(path)
    if path.suffix.lower() == ".mid" or path.suffix.lower() == ".midi":
        return _load_midi(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path, verbose=verbose)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches, params):
    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        return 0.0, 0.0, 0.0
    ref_hz = np.array([pretty_midi.note_number_to_hz(p) for p in ref_pitches])
    est_hz = np.array([pretty_midi.note_number_to_hz(p) for p in est_pitches])
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, est_intervals, est_hz, **params
    )
    return p, r, f


def f1_standard(ref_intervals, ref_pitches, est_intervals, est_pitches):
    return f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches, STANDARD_PARAMS)


def f1_strict(ref_intervals, ref_pitches, est_intervals, est_pitches):
    return f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches, STRICT_PARAMS)


def f1_onset_only(ref_intervals, ref_pitches, est_intervals, est_pitches):
    return f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches, ONSET_ONLY_PARAMS)


def _match(ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio, offset_min_tolerance):
    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        return []
    ref_hz = np.array([pretty_midi.note_number_to_hz(p) for p in ref_pitches])
    est_hz = np.array([pretty_midi.note_number_to_hz(p) for p in est_pitches])
    return mir_eval.transcription.match_notes(
        ref_intervals, ref_hz, est_intervals, est_hz,
        onset_tolerance=0.05,
        pitch_tolerance=50.0,
        offset_ratio=offset_ratio,
        offset_min_tolerance=offset_min_tolerance,
    )


def deviation_mae(ref_intervals, ref_pitches_int, ref_pitches_frac,
                  est_intervals, est_pitches_int, est_pitches_frac):
    """Onset / offset / pitch MAE over matched notes.

    Returns dict with onset_mae_ms, offset_mae_ms, pitch_mae_cents,
    n_match_std (used for onset & pitch), n_match_offset (used for offset).
    Uses two match calls per Plan agent recommendation: one without offset
    constraint for onset/pitch (largest valid set), one with standard offset
    for offset MAE.
    """
    matches_no_off = _match(
        ref_intervals, ref_pitches_int, est_intervals, est_pitches_int,
        offset_ratio=None, offset_min_tolerance=0.05,
    )
    matches_std = _match(
        ref_intervals, ref_pitches_int, est_intervals, est_pitches_int,
        offset_ratio=0.2, offset_min_tolerance=0.05,
    )

    onset_mae_ms = np.nan
    pitch_mae_cents = np.nan
    if matches_no_off:
        onsets = [
            abs(est_intervals[j, 0] - ref_intervals[i, 0]) for i, j in matches_no_off
        ]
        onset_mae_ms = float(np.mean(onsets) * 1000.0)
        pitches = [
            abs(est_pitches_frac[j] - ref_pitches_frac[i]) for i, j in matches_no_off
        ]
        pitch_mae_cents = float(np.mean(pitches) * 100.0)

    offset_mae_ms = np.nan
    if matches_std:
        offsets = [
            abs(est_intervals[j, 1] - ref_intervals[i, 1]) for i, j in matches_std
        ]
        offset_mae_ms = float(np.mean(offsets) * 1000.0)

    return {
        "onset_mae_ms": onset_mae_ms,
        "offset_mae_ms": offset_mae_ms,
        "pitch_mae_cents": pitch_mae_cents,
        "n_match_std": len(matches_no_off),
        "n_match_offset": len(matches_std),
    }


def evaluate_pair(ref_path, est_path, verbose=False):
    """Run standard F1, strict F1, and the three MAEs for one (ref, est) pair."""
    ref_int, ref_pi, ref_pf = load_notes(ref_path, verbose=verbose)
    est_int, est_pi, est_pf = load_notes(est_path, verbose=verbose)

    p_std, r_std, f_std = f1_standard(ref_int, ref_pi, est_int, est_pi)
    p_str, r_str, f_str = f1_strict(ref_int, ref_pi, est_int, est_pi)
    p_on,  r_on,  f_on  = f1_onset_only(ref_int, ref_pi, est_int, est_pi)
    mae = deviation_mae(ref_int, ref_pi, ref_pf, est_int, est_pi, est_pf)

    return {
        "n_ref": len(ref_int),
        "n_est": len(est_int),
        "P_onset": p_on, "R_onset": r_on, "F_onset": f_on,
        "P_std": p_std, "R_std": r_std, "F_std": f_std,
        "P_strict": p_str, "R_strict": r_str, "F_strict": f_str,
        **mae,
    }


# ----------------------------------------------------------------------
# Auto-discovery: pair every *_truther*.csv with its raw .mid + stage CSVs
# ----------------------------------------------------------------------

TRUTH_TOKENS = ("groundtruther", "truther", "truer", "truth")
PITCH_TOKENS = ("pitch", "newestpitch")
OFFSET_TOKENS = ("offset", "newestoffset")


def _pick_latest(paths):
    """Pick the most recently exported file from a sorted list. Filenames in this
    repo carry timestamps like '28-Apr-2026 06-30-32', so the lexicographically
    last entry is the most recent — but only after grouping by token, not across.
    Simpler approach: pick the one with the greatest mtime."""
    if not paths:
        return None
    return str(max(paths, key=lambda p: p.stat().st_mtime))


def discover(directory, raw_fallback_dir=None):
    """Walk a directory, return a list of {tune, truth, stages: {raw, +pitch, +offset}} dicts.

    Naming conventions supported (any of these tokens is recognised):
      truth:  <tune>_truther*.csv  | <tune>_truer*.csv  | <tune>_truth*.csv
      raw:    <tune>_*.mid         (optional; if absent, raw stage is skipped)
      pitch:  <tune>_*_pitch*.csv  | <tune>_*_newestpitch*.csv
      offset: <tune>_*_offset*.csv | <tune>_*_newestoffset*.csv

    Tune name is whatever precedes the truth-token suffix in the truth filename.
    Multiple stage CSVs for the same tune resolve to the most recently modified
    file. Stages with no file resolve to None and are skipped during evaluation.

    If `raw_fallback_dir` is provided, missing raw .mid files are also looked up
    there. Useful when a refined-stage export omits the unchanged raw output.
    """
    from pathlib import Path
    directory = Path(directory)
    fallback = Path(raw_fallback_dir) if raw_fallback_dir else None
    tunes = []

    # Find truth files using any of the truth tokens
    truth_files = []
    seen = set()
    for tok in TRUTH_TOKENS:
        for p in directory.glob(f"*_{tok}*.csv"):
            if p not in seen:
                truth_files.append(p)
                seen.add(p)
    truth_files.sort()

    for truth in truth_files:
        # Tune name is the part before the first matching truth token
        tune = truth.name
        for tok in TRUTH_TOKENS:
            marker = f"_{tok}"
            if marker in tune:
                tune = tune.split(marker)[0]
                break

        # Raw .mid — try directory first, then fallback. Pick shortest name (no stage suffix).
        raw_candidates = sorted(directory.glob(f"{tune}_*.mid"), key=lambda p: len(p.name))
        if not raw_candidates and fallback:
            raw_candidates = sorted(fallback.glob(f"{tune}_*.mid"), key=lambda p: len(p.name))
        raw = str(raw_candidates[0]) if raw_candidates else None

        # Stage CSVs: union over all stage tokens, scoped to this tune
        pitch_csvs = []
        offset_csvs = []
        for tok in PITCH_TOKENS:
            pitch_csvs.extend(directory.glob(f"{tune}_*_{tok}*.csv"))
        for tok in OFFSET_TOKENS:
            offset_csvs.extend(directory.glob(f"{tune}_*_{tok}*.csv"))
        # Filter out truth files that may have been picked up by overlapping patterns
        pitch_csvs = [p for p in pitch_csvs if not any(t in p.name for t in (f"_{x}" for x in TRUTH_TOKENS))]
        offset_csvs = [p for p in offset_csvs if not any(t in p.name for t in (f"_{x}" for x in TRUTH_TOKENS))]

        tunes.append({
            "tune": tune,
            "truth": str(truth),
            "stages": {
                "raw": raw,
                "+pitch": _pick_latest(pitch_csvs),
                "+offset": _pick_latest(offset_csvs),
            },
        })
    return tunes


def discover_split(test_dir, model_dir=None, refined_dir=None):
    """Discover tunes from a `<test>/GT/<tune>.mid` + `<test>/<model>/<tune>_transcribed_*.mid` layout.

    Used for evaluating a held-out test split where every tune has a MIDI ground
    truth (no fractional pitch) and a model-emitted MIDI prediction.

    test_dir/
      GT/<tune>.mid                                   -> truth
      <model>/<tune>_transcribed_*.mid                -> raw stage
      <model>/<tune>.mid                              -> raw stage (alt naming)

    refined_dir (optional): pull `<tune>*_pitch*.csv` and `<tune>*_offset*.csv`
    as +pitch / +offset stages. Tunes without refined files just get None for
    those stages and are skipped during evaluation.
    """
    from pathlib import Path
    test_dir = Path(test_dir)
    gt_dir = test_dir / "GT"
    if not gt_dir.exists():
        raise FileNotFoundError(f"No GT/ subdir in {test_dir}")
    if model_dir is None:
        candidates = [p for p in test_dir.iterdir() if p.is_dir() and p.name != "GT"]
        if not candidates:
            raise FileNotFoundError(f"No model subdir in {test_dir}")
        model_path = candidates[0]
    else:
        model_path = (test_dir / model_dir) if not Path(model_dir).is_absolute() else Path(model_dir)

    refined = Path(refined_dir) if refined_dir else None
    tunes = []
    for gt_path in sorted(gt_dir.glob("*.mid")):
        tune = gt_path.stem
        # Raw: prefer "<tune>_transcribed*.mid", fall back to "<tune>.mid"
        cand = sorted(model_path.glob(f"{tune}_transcribed*.mid"))
        if not cand:
            cand = sorted(model_path.glob(f"{tune}.mid"))
        raw = str(cand[0]) if cand else None

        # Truth: prefer a `<tune>_groundtruther/_truther/_truer*.csv` from the
        # refined dir (richer fractional pitch annotation). Fall back to the
        # GT/<tune>.mid otherwise.
        truth = str(gt_path)
        if refined is not None:
            for tok in TRUTH_TOKENS:
                cand_t = sorted(refined.glob(f"{tune}_{tok}*.csv"))
                if cand_t:
                    truth = str(_pick_latest(cand_t))
                    break

        pitch = offset = None
        if refined is not None:
            pc = list(refined.glob(f"{tune}_*pitch*.csv"))
            oc = list(refined.glob(f"{tune}_*offset*.csv"))
            pc = [p for p in pc if not any(f"_{t}" in p.name for t in TRUTH_TOKENS)]
            oc = [p for p in oc if not any(f"_{t}" in p.name for t in TRUTH_TOKENS)]
            pitch = _pick_latest(pc)
            offset = _pick_latest(oc)

        tunes.append({
            "tune": tune,
            "truth": truth,
            "stages": {"raw": raw, "+pitch": pitch, "+offset": offset},
        })
    return tunes


def evaluate_split(test_dir, model_dir=None, refined_dir=None,
                    stages=("raw", "+pitch", "+offset"), exclude=()):
    """evaluate_pair across every (tune, stage) found by discover_split().

    `exclude` is an iterable of substrings; any tune whose name contains any of
    them is skipped. Useful for stripping the emotional variants
    (`exclude=('_angry', '_happy', '_sad', '_tender')`).
    """
    rows = []
    for entry in discover_split(test_dir, model_dir=model_dir, refined_dir=refined_dir):
        if any(s in entry["tune"] for s in exclude):
            continue
        for stage in stages:
            est = entry["stages"].get(stage)
            if est is None:
                continue
            r = evaluate_pair(entry["truth"], est)
            rows.append({"tune": entry["tune"], "stage": stage, **r})
    return rows


def evaluate_directory(directory, stages=("raw", "+pitch", "+offset"), raw_fallback_dir=None):
    """Run evaluate_pair on every (tune, stage) found by discover()."""
    rows = []
    for entry in discover(directory, raw_fallback_dir=raw_fallback_dir):
        for stage in stages:
            est = entry["stages"].get(stage)
            if est is None:
                continue
            r = evaluate_pair(entry["truth"], est)
            rows.append({"tune": entry["tune"], "stage": stage, **r})
    return rows


# ----------------------------------------------------------------------
# Diagnostics: detect identical stages, pitch bias, and duration floors
# ----------------------------------------------------------------------

def diagnose_stage(ref_path, est_path):
    """Surface silently bundled transformations between truth and a stage.

    Returns a dict with:
      identical_to_prev: True if (onset, offset, onpitch) match a comparison file
                         (caller passes this in via diagnose_identical instead).
      pitch_bias_cents:   mean signed deviation est-truth over matched notes.
      pitch_p95_cents:    95th percentile of |est-truth| in cents.
      duration_floor_ms:  smallest duration in est (suggests a hard floor when
                          a large fraction sits at exactly that value).
      duration_floor_count: number of est notes within 0.1 ms of the floor.
      n_est:              total est notes.
    """
    import numpy as np
    import mir_eval, pretty_midi
    ref_int, ref_pi, ref_pf = load_notes(ref_path)
    est_int, est_pi, est_pf = load_notes(est_path)
    ref_hz = np.array([pretty_midi.note_number_to_hz(p) for p in ref_pi])
    est_hz = np.array([pretty_midi.note_number_to_hz(p) for p in est_pi])
    matches = mir_eval.transcription.match_notes(
        ref_int, ref_hz, est_int, est_hz,
        onset_tolerance=0.05, pitch_tolerance=200.0,
        offset_ratio=None, offset_min_tolerance=0.05)
    if matches:
        diffs = np.array([est_pf[j] - ref_pf[i] for i, j in matches]) * 100
        bias = float(diffs.mean())
        p95 = float(np.percentile(np.abs(diffs), 95))
    else:
        bias, p95 = float("nan"), float("nan")
    durs_ms = (est_int[:, 1] - est_int[:, 0]) * 1000
    floor_ms = float(durs_ms.min()) if len(durs_ms) else float("nan")
    floor_count = int((np.abs(durs_ms - floor_ms) < 0.1).sum())
    return {
        "pitch_bias_cents": bias,
        "pitch_p95_cents": p95,
        "duration_floor_ms": floor_ms,
        "duration_floor_count": floor_count,
        "n_est": len(est_int),
    }


def per_note_diagnosis(ref_path, est_path):
    """Per-note diagnostic comparing one (truth, estimate) pair.

    Returns a DataFrame with one row per truth note plus one row per unmatched
    est note (false positives). Columns:
      truth_idx, est_idx, truth_onset, truth_offset, truth_pitch,
      est_onset, est_offset, est_pitch,
      onset_diff_ms, offset_diff_ms, pitch_diff_cents,
      status: one of
        matched_strict       — passes onset, pitch, AND strict offset (5%/25ms)
        matched_std          — passes onset and pitch tolerances; offset may not
        unmatched_pitch      — best candidate within onset tol but pitch > 50c
        unmatched_offset     — like matched_std but failed strict offset
        unmatched_pitch+onset — both onset and pitch out of tolerance
        missed               — no est note within loose-onset (100ms) range
        extra                — est note that didn't match any truth (false +ve)

    Useful when the post-processor's author asks "show me the bad notes".
    Sort by abs(pitch_diff_cents) desc to surface the worst pitch offenders;
    sort by abs(offset_diff_ms) for offset offenders.
    """
    import numpy as np
    import mir_eval, pretty_midi
    import pandas as pd
    ref_int, ref_pi, ref_pf = load_notes(ref_path)
    est_int, est_pi, est_pf = load_notes(est_path)
    ref_hz = np.array([pretty_midi.note_number_to_hz(p) for p in ref_pi])
    est_hz = np.array([pretty_midi.note_number_to_hz(p) for p in est_pi])

    M_std = dict(mir_eval.transcription.match_notes(
        ref_int, ref_hz, est_int, est_hz,
        onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=None, offset_min_tolerance=0.05))
    M_strict = dict(mir_eval.transcription.match_notes(
        ref_int, ref_hz, est_int, est_hz,
        onset_tolerance=0.05, pitch_tolerance=50.0,
        offset_ratio=0.05, offset_min_tolerance=0.025))
    # Loose match: onset 100 ms, pitch 4 semitones — gets a candidate for almost every truth
    M_loose = dict(mir_eval.transcription.match_notes(
        ref_int, ref_hz, est_int, est_hz,
        onset_tolerance=0.1, pitch_tolerance=400.0,
        offset_ratio=None, offset_min_tolerance=0.1))

    rows = []
    matched_est = set()
    for i in range(len(ref_int)):
        if i in M_std:
            j = M_std[i]
            status = "matched_strict" if i in M_strict else (
                "matched_std" if (i in M_std) else "matched_std")
            # If matched in std but not strict, status is matched_std + offset miss
            if i not in M_strict:
                status = "unmatched_offset"
            else:
                status = "matched_strict"
        elif i in M_loose:
            j = M_loose[i]
            onset_off = abs(est_int[j, 0] - ref_int[i, 0])
            cents_off = abs(1200 * np.log2(est_hz[j] / ref_hz[i]))
            reasons = []
            if onset_off > 0.05:
                reasons.append("onset")
            if cents_off > 50:
                reasons.append("pitch")
            status = "unmatched_" + "+".join(reasons) if reasons else "unmatched_other"
        else:
            j = None
            status = "missed"

        if j is not None:
            matched_est.add(j)
            rows.append({
                "truth_idx": i, "est_idx": j,
                "truth_onset": float(ref_int[i, 0]),
                "truth_offset": float(ref_int[i, 1]),
                "truth_pitch": float(ref_pf[i]),
                "est_onset": float(est_int[j, 0]),
                "est_offset": float(est_int[j, 1]),
                "est_pitch": float(est_pf[j]),
                "onset_diff_ms": float((est_int[j, 0] - ref_int[i, 0]) * 1000),
                "offset_diff_ms": float((est_int[j, 1] - ref_int[i, 1]) * 1000),
                "pitch_diff_cents": float((est_pf[j] - ref_pf[i]) * 100.0),
                "status": status,
            })
        else:
            rows.append({
                "truth_idx": i, "est_idx": None,
                "truth_onset": float(ref_int[i, 0]),
                "truth_offset": float(ref_int[i, 1]),
                "truth_pitch": float(ref_pf[i]),
                "est_onset": None, "est_offset": None, "est_pitch": None,
                "onset_diff_ms": None, "offset_diff_ms": None, "pitch_diff_cents": None,
                "status": "missed",
            })

    for j in range(len(est_int)):
        if j not in matched_est:
            rows.append({
                "truth_idx": None, "est_idx": j,
                "truth_onset": None, "truth_offset": None, "truth_pitch": None,
                "est_onset": float(est_int[j, 0]),
                "est_offset": float(est_int[j, 1]),
                "est_pitch": float(est_pf[j]),
                "onset_diff_ms": None, "offset_diff_ms": None, "pitch_diff_cents": None,
                "status": "extra",
            })

    return pd.DataFrame(rows)


def diagnose_identical(p1, p2, cols=("onset", "offset", "onpitch")):
    """Are two CSVs byte-identical on the supplied columns? Returns bool or None
    if either file isn't a CSV."""
    from pathlib import Path
    if Path(p1).suffix.lower() != ".csv" or Path(p2).suffix.lower() != ".csv":
        return None
    a = pd.read_csv(p1)[list(cols)].dropna()
    b = pd.read_csv(p2)[list(cols)].dropna()
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a.values, b.values, atol=1e-9))

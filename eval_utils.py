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
    mae = deviation_mae(ref_int, ref_pi, ref_pf, est_int, est_pi, est_pf)

    return {
        "n_ref": len(ref_int),
        "n_est": len(est_int),
        "P_std": p_std, "R_std": r_std, "F_std": f_std,
        "P_strict": p_str, "R_strict": r_str, "F_strict": f_str,
        **mae,
    }

import argparse
import math
import traceback
from typing import Dict, Tuple

import parselmouth
import pandas as pd


VOWELS = {"a", "e", "ɛ", "i", "o", "ø", "u", "y"}
N_FORMANTS = 5


def is_vowel(phoneme: str) -> bool:
    return phoneme.lower().strip() in VOWELS


def _max_formant(gender: str) -> float:
    return 5000.0 if gender.lower() == "f" else 4500.0


def _safe_formant(formant_obj, fn: int, t: float) -> float:
    try:
        v = formant_obj.get_value_at_time(fn, t)
        return v if (v is not None and not math.isnan(v)) else float("nan")
    except Exception:
        return float("nan")


def extract_f1_f2(
    snd: parselmouth.Sound,
    gender: str,
    measure_point: float = 0.5,
) -> Tuple[float, float]:
    """
    Extract F1 and F2 at a single relative time point within the segment.
    measure_point: 0.25 | 0.5 | 0.75  (fraction of segment duration)
    Returns (F1, F2) in Hz.
    """
    max_formant = _max_formant(gender)
    formant = snd.to_formant_burg(
        maximum_formant=max_formant,
        max_number_of_formants=N_FORMANTS,
    )
    t = formant.start_time + formant.duration * measure_point
    f1 = _safe_formant(formant, 1, t)
    f2 = _safe_formant(formant, 2, t)
    return f1, f2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure-point", "-m",
                        choices=["0.25", "0.5", "0.75"], default="0.5")
    parser.add_argument("--input",  "-i", default="./tables/metadata.csv")
    parser.add_argument("--output", "-o", default="./reps/features_acoustic.csv")
    parser.add_argument("--missing",         default="./tables/missing.csv")
    parser.add_argument("--missing-summary", default="./tables/missing_summary.csv")
    args = parser.parse_args()

    measure_pt = float(args.measure_point)

    df = pd.read_csv(args.input).fillna("")

    f1_col    = []
    f2_col    = []
    error_log = []

    sound_cache: Dict[str, parselmouth.Sound] = {}

    for _, row in df.iterrows():
        phone  = row["phoneme"]
        wav    = row["wav_path"]
        on     = float(row["onset"])
        off    = float(row["offset"])
        gender = str(row["Gender"])

        # non-phoneme rows (silences, empty labels) — skip extraction
        if phone == "":
            f1_col.append(float("nan"))
            f2_col.append(float("nan"))
            continue

        try:
            if wav not in sound_cache:
                sound_cache[wav] = parselmouth.Sound(wav)
            segment = sound_cache[wav].extract_part(
                from_time=on,
                to_time=off,
                preserve_times=False,
            )
            f1, f2 = extract_f1_f2(segment, gender, measure_pt)
            f1_col.append(f1)
            f2_col.append(f2)

            if math.isnan(f1) or math.isnan(f2):
                error_log.append({
                    "wav": wav, "phoneme": phone,
                    "error": "F1 or F2 is NaN (formant tracker failure)",
                })

        except Exception as e:
            f1_col.append(float("nan"))
            f2_col.append(float("nan"))
            error_log.append({
                "wav": wav, "phoneme": phone,
                "error": f"{type(e).__name__}: {e}",
            })

    df["F1"] = f1_col
    df["F2"] = f2_col
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows → {args.output}")

    # ── missing-value report ──────────────────────────────────────────────────
    err_df = pd.DataFrame(error_log)
    err_df.to_csv(args.missing, index=False)

    # per phoneme
    total   = df.groupby("phoneme")["F1"].size().rename("total")
    missing = df[df["F1"].isna()].groupby("phoneme").size().rename("missing")
    summary = pd.merge(total, missing, left_index=True,
                       right_index=True, how="left").fillna(0)
    summary["missing_pct"] = (summary["missing"] / summary["total"] * 100).round(1)
    summary["missing"] = summary["missing"].astype(int)

    # per phoneme × L1 group
    group_total   = df.groupby(["phoneme", "L1"])["F1"].size().rename("total")
    group_missing = (df[df["F1"].isna()]
                     .groupby(["phoneme", "L1"]).size().rename("missing"))
    group_summary = pd.merge(group_total, group_missing,
                             left_index=True, right_index=True,
                             how="left").fillna(0).reset_index()
    group_summary["missing_pct"] = (
        group_summary["missing"] / group_summary["total"] * 100
    ).round(1)
    group_summary["missing"] = group_summary["missing"].astype(int)

    # write both to one CSV with a source column
    summary_out = summary.reset_index()
    summary_out["L1"] = "all"
    combined = pd.concat([summary_out, group_summary], ignore_index=True)
    combined.to_csv(args.missing_summary, index=False)

    print(f"Missing report → {args.missing}")
    print(f"Missing summary → {args.missing_summary}")
    print(f"\nOverall missing F1/F2 per phoneme:")
    print(summary[["total", "missing", "missing_pct"]].to_string())

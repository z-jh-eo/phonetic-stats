import argparse
import math
import os
import traceback
from typing import Dict, Tuple

import parselmouth
import pandas as pd


VOWELS = {"a", "e", "ɛ", "i", "o", "ø", "u", "y"}
VOICED = {"b", "d", "g", "z", "v", "ʒ", "m", "n", 
          "ʁ", "l", "w", "j",
          "a", "e", "ɛ", "i", "o", "ø", "u", "y"}
FRICATIVES = {"f", "v", "s", "z", "ʃ", "ʒ", "ʁ"}

PITCH_FLOOR = 100.0
PITCH_CEILING = 600.0
N_FORMANTS = 5


def is_vowel(phoneme: str) -> bool:
    return phoneme.lower().strip() in VOWELS


def is_voiced(phoneme: str) -> bool:
    return phoneme.lower().strip() in VOICED


def is_fricative(phoneme: str) -> bool:
    return phoneme.lower().strip() in FRICATIVES


def _max_formant(gender: str) -> float:
    return 5000.0 if gender.lower() == "f" else 4500.0


def _safe_f0(pitch_obj, t: float) -> float:
    try:
        v = pitch_obj.get_value_at_time(time=t)
        return v if (v is not None and not math.isnan(v)) else float("nan")
    except Exception:
        return float("nan")


def _safe_formant(formant_obj, fn: int, t: float) -> float:
    try:
        v = formant_obj.get_value_at_time(fn, t)
        return v if (v is not None and not math.isnan(v)) else float("nan")
    except Exception:
        return float("nan")


def _safe_scg(spectrum_obj) -> float:
    try:
        v = spectrum_obj.get_center_of_gravity()
        return v if (v is not None and not math.isnan(v)) else float("nan")
    except Exception:
        return float("nan")


def _measure_at(pitch_obj, formant_obj, t_p: float, t_f: float, voiced: bool) -> Tuple[float, float, float, float]:
    f0 = _safe_f0(pitch_obj, t_p) if voiced else float("nan")
    f1 = _safe_formant(formant_obj, 1, t_f)
    f2 = _safe_formant(formant_obj, 2, t_f)
    f3 = _safe_formant(formant_obj, 3, t_f)
    return (f0, f1, f2, f3)


def extract_feats(
    snd: parselmouth.Sound,
    gender: str,
    voiced: bool,
    is_long_v: bool = False,
    is_fric: bool = False,
) -> Tuple[float, ...]:
    max_formant = _max_formant(gender)

    pitch = snd.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    formant = snd.to_formant_burg(maximum_formant=max_formant, max_number_of_formants=N_FORMANTS)
    d = snd.duration

    t_p_mid = pitch.start_time + (pitch.duration / 2)
    t_f_mid = formant.start_time + (formant.duration / 2)
    mid = _measure_at(pitch, formant, t_p_mid, t_f_mid, voiced)

    nan4 = (float("nan"), float("nan"), float("nan"), float("nan"))

    if is_long_v:
        t_p_25 = pitch.start_time + (pitch.duration * 0.25)
        t_p_75 = pitch.start_time + (pitch.duration * 0.75)
        t_f_25 = formant.start_time + (formant.duration * 0.25)
        t_f_75 = formant.start_time + (formant.duration * 0.75)

        p25 = _measure_at(pitch, formant, t_p_25, t_f_25, voiced)
        p75 = _measure_at(pitch, formant, t_p_75, t_f_75, voiced)
    else:
        p25, p75 = nan4, nan4

    scg = _safe_scg(snd.to_spectrum()) if is_fric else float("nan")

    return (
        d,
        # midpoint
        mid[0], mid[1], mid[2], mid[3],
        # trajectory points
        p25[0], p25[1], p25[2], p25[3],
        p75[0], p75[1], p75[2], p75[3],
        scg
    )


def extract_signal_features(
    phoneme: str,
    audio_path: str,
    onset: float,
    offset: float,
    gender: str,
) -> Dict:
    voiced = is_voiced(phoneme)
    dur_ms = (offset - onset) * 1000.0
    is_long_v = is_vowel(phoneme) and dur_ms >= 79.0
    is_fric = is_fricative(phoneme)

    try:
        sound = parselmouth.Sound(audio_path).extract_part(
            from_time=onset,
            to_time=offset,
            preserve_times=False,
        )
        feats = extract_feats(sound, gender, voiced, is_long_v, is_fric)
        res = {
            "ok": True,
            "err": "",
            "is_voiced": voiced,
            "feats": feats,
        }
    except Exception as e:
        nan = float("nan")
        res = {
            "ok": False,
            "err": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(),
            "is_voiced": voiced,
            "feats": (
                nan,
                nan, nan, nan, nan,
                nan, nan, nan, nan,
                nan, nan, nan, nan,
                nan
            )
        }
    return res


def _pick_f1_f2(feats: Tuple[float, ...], measure_point: str) -> Tuple[float, float]:
    # indices: 2=f1_mid,3=f2_mid, 6=f1_25,7=f2_25, 10=f1_75,11=f2_75
    f1_mid, f2_mid = feats[2], feats[3]

    if measure_point == "0.5":
        return f1_mid, f2_mid
    if measure_point == "0.25":
        f1 = feats[6]
        f2 = feats[7]
    elif measure_point == "0.75":
        f1 = feats[10]
        f2 = feats[11]
    else:
        f1 = f2 = float("nan")

    f1 = f1 if not math.isnan(f1) else f1_mid
    f2 = f2 if not math.isnan(f2) else f2_mid
    return f1, f2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure-point", "-m", choices=["0.25", "0.5", "0.75"], default="0.5")
    parser.add_argument("--input", "-i", default="./tables/metadata.csv")
    parser.add_argument("--output", "-o", default="./reps/features_acoustic.csv")
    parser.add_argument("--missing", "-e", default="./tables/missing.csv")
    parser.add_argument("--missing-summary", "-s", default="./tables/missing_summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input).fillna("")

    features = []
    f1 = []
    f2 = []
    error_log = []

    # Cache loaded sounds by file path
    sound_cache: Dict[str, parselmouth.Sound] = {}

    for _, row in df.iterrows():
        phone = row["phoneme"]
        wav = row["wav_path"]
        on = row["onset"]
        off = row["offset"]
        gender = row["Gender"]

        if phone == "":
            features.append(float("nan"))
            f1.append(float("nan"))
            f2.append(float("nan"))
            continue

        try:
            if wav not in sound_cache:
                sound_cache[wav] = parselmouth.Sound(wav)
            sound = sound_cache[wav].extract_part(
                from_time=on,
                to_time=off,
                preserve_times=False,
            )
            res = {
                "ok": True,
                "err": "",
                "is_voiced": is_voiced(phone),
                "feats": extract_feats(
                    sound,
                    gender,
                    is_voiced(phone),
                    is_vowel(phone) and (off - on) * 1000.0 >= 79.0,
                    is_fricative(phone),
                )
            }
        except Exception as e:
            nan = float("nan")
            res = {
                "ok": False,
                "err": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(),
                "is_voiced": is_voiced(phone),
                "feats": (nan,) * 14
            }

        if res["ok"]:
            features.append(res["feats"])
            f1_val, f2_val = _pick_f1_f2(res["feats"], args.measure_point)
            f1.append(f1_val)
            f2.append(f2_val)

            if res["is_voiced"] and math.isnan(res["feats"][1]):
                error_log.append({
                    "wav": wav,
                    "phoneme": phone,
                    "error": "Voiced phoneme but f0 is NaN"
                })
        else:
            features.append(float("nan"))
            f1.append(float("nan"))
            f2.append(float("nan"))
            error_log.append({
                "wav": wav,
                "phoneme": phone,
                "error": res["err"]
            })

    print(len(df))
    print(len(features))
    print(len(f1))
    df["signal_rep"] = features
    df["F1"] = f1
    df["F2"] = f2

    os.makedirs("./reps", exist_ok=True)

    df.to_csv(args.output, index=False)

    err = pd.DataFrame(error_log)
    err.to_csv(args.missing, index=False)

    full_count = df.groupby("phoneme")["signal_rep"].size().rename("total")
    missing_ct = df[df["signal_rep"].isna()].groupby("phoneme").size().rename("missing")
    ct = pd.merge(full_count, missing_ct, left_index=True, right_index=True, how="left")
    ct["missing_pct"] = (ct["missing"] / ct["total"] * 100).round(1)

    ct.to_csv(args.missing_summary, index=False)
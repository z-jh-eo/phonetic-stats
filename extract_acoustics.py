import math
import parselmouth
import pandas as pd


def is_vowel(phoneme: str) -> bool:
    return phoneme.lower().strip() in "aeɛioøuy"

def is_voiced(phoneme: str) -> bool:
    return phoneme.lower().strip() in "bdgzvʒmnʁlwjaeɛioøuy"

def is_fricative(phoneme: str) -> bool:
    return phoneme.lower().strip() in "fvszʃʒʁ"


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

def _measure_at(pitch_obj, formant_obj, t_p: float, t_f: float, voiced: bool) -> tuple:
    f0 = _safe_f0(pitch_obj, t_p) if voiced else float("nan")
    f1 = _safe_formant(formant_obj, 1, t_f)
    f2 = _safe_formant(formant_obj, 2, t_f)
    f3 = _safe_formant(formant_obj, 3, t_f)
    return (f0, f1, f2, f3)


PITCH_FLOOR = 100.0
PITCH_CEILING = 600.0
N_FORMANTS = 5


def extract_feats(
        snd: parselmouth.Sound,
        gender: str,
        is_voiced: bool,
        is_long_v: bool=False,
        is_fricative: bool=False,
) -> tuple:
    
    max_formant = _max_formant(gender)

    pitch = snd.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    formant = snd.to_formant_burg(maximum_formant=max_formant, max_number_of_formants=N_FORMANTS)
    d = snd.duration

    t_p_mid = pitch.start_time + (pitch.duration / 2)
    t_f_mid = formant.start_time + (formant.duration / 2)
    mid = _measure_at(pitch, formant, t_p_mid, t_f_mid, is_voiced)

    nan4 = (float("nan"), float("nan"), float("nan"), float("nan")) 

    if is_long_v:
        t_p_25 = pitch.start_time + (pitch.duration * 0.25)
        t_p_75 = pitch.start_time + (pitch.duration * 0.75)
        t_f_25 = formant.start_time + (formant.duration * 0.25)
        t_f_75 = formant.start_time + (formant.duration * 0.75)

        p25 = _measure_at(pitch, formant, t_p_25, t_f_25, is_voiced)
        p75 = _measure_at(pitch, formant, t_p_75, t_f_75, is_voiced)
    else:
        p25, p75 = nan4, nan4
    
    if is_fricative:
        scg = _safe_scg(snd.to_spectrum())
    else:
        scg = float("nan")

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
) -> dict:
    
    voiced    = is_voiced(phoneme)
    dur_ms    = (offset - onset) * 999.0
    is_long_v = is_vowel(phoneme) and dur_ms >= 79.0
    is_fric   = is_fricative(phoneme)
    
    try:
        sound = parselmouth.Sound(audio_path)\
                           .extract_part(
                               from_time=onset, to_time=offset,
                               preserve_times=False,   # re-zero timestamps inside the snippet
                            )
        feats = extract_feats(sound, gender, voiced, is_long_v, is_fric)
        res = {
            "ok": True,
            "err": "",
            "is_voiced": voiced,
            "feats": feats
        }
    
    except Exception as e:
        nan = float("nan")
        res = {
            "ok": False,
            "err": str(e),
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


df = pd.read_csv("./metadata.csv").fillna("")

features = []
missing_repo = []
for _, row in df.iterrows():
    phone = row["phoneme"]
    wav = row["wav_path"]
    on = row["onset"]
    off = row["offset"]
    gender = row["Gender"]
    if phone == "":
        features.append(float("nan"))
    else:
        res = extract_signal_features(phone, wav, on, off, gender)
        if res["ok"]:
            features.append(res["feats"])
            if res["is_voiced"] and res["feats"][1] == float("nan"):
                missing_repo.append({
                    "wav": wav,
                    "phoneme": phone,
                    "error": "Voiced phoneme but f0 is NaN"
                })
        else:
            #print(f"Error processing {wav} ({phone}): {res["err"]}")
            features.append(float("nan"))
            missing_repo.append({
                "wav": wav,
                "phoneme": phone,
                "error": res["err"]
            })
df["signal_rep"] = features
df.to_csv("./features_acoustic.csv", index=False)

err = pd.DataFrame(missing_repo)
err.to_csv("./missing.csv", index=False)

full_count = df.groupby("phoneme")["signal_rep"].size().rename("total")
missing_ct = df[df["signal_rep"].isna()].groupby("phoneme").size().rename("missing")
ct = pd.merge(full_count, missing_ct, left_index=True, right_index=True, how="left")
ct["missing_pct"] = (ct["missing"] / ct["total"] * 100).round(1)

ct.to_csv("./missing_summary.csv")
#print(ct)
#print(df.head())
import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from normalise import is_vowel, lobanov_norm

def decompose_variance(df: pd.DataFrame, phoneme: str) -> dict:
    sub = df[df["phoneme"] == phoneme].dropna(subset=["F1_normed", "spk"])
    if sub.empty or sub["spk"].nunique() < 2:
        return {
            "phoneme": phoneme,
            "n": len(sub),
            "n_speakers": sub["spk"].nunique(),
            "var_total": np.nan,
            "var_inter_speaker": np.nan,
            "var_intra_speaker": np.nan,
            "var_residual": np.nan
        }

    # total variance
    var_total = np.nanvar(sub["F1_normed"], ddof=1)

    # mixed model: random intercept for speaker
    try:
        md = smf.mixedlm("F1_normed ~ 1", sub, groups=sub["spk"])
        mdf = md.fit(reml=True)
        var_inter = float(mdf.cov_re.iloc[0, 0])
        var_resid = float(mdf.scale)
    except Exception:
        var_inter = np.nan
        var_resid = np.nan

    # intra-speaker variance: mean of within-speaker variances
    within = sub.groupby("spk")["F1_normed"].var(ddof=1)
    var_intra = float(within.mean()) if len(within) > 0 else np.nan

    return {
        "phoneme": phoneme,
        "n": len(sub),
        "n_speakers": sub["spk"].nunique(),
        "var_total": var_total,
        "var_inter_speaker": var_inter,
        "var_intra_speaker": var_intra,
        "var_residual": var_resid
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./reps/features_acoustic.csv")
    parser.add_argument("--output", default="./tables/variance_decomposition.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["phoneme"] = df["phoneme"].fillna("")
    df = lobanov_norm(df)

    vowels = sorted(df[df["phoneme"].apply(is_vowel)]["phoneme"].unique().tolist())

    rows = [decompose_variance(df, p) for p in vowels]
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
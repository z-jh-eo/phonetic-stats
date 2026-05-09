import argparse
import numpy as np
import pandas as pd
from scipy.stats import shapiro, wilcoxon, ttest_rel
from statsmodels.stats.multitest import multipletests
from normalise import is_vowel, lobanov_norm


def test_gender_one(phoneme: str, formant: str,
                    spk_means: pd.DataFrame) -> dict:
    """
    Paired test at the speaker level for one (phoneme, formant).

    spk_means has columns: [spk, Gender, <formant>]
    One row per speaker = their mean normalised formant for this phoneme.
    Pairing is done by matching speakers across genders — since this is a
    between-subjects factor (each speaker has one gender), "paired" here means
    we use the speaker mean as the unit of observation and apply:
      - Wilcoxon signed-rank  if n_per_group is small (< 20) or non-normal
      - Paired t-test         if both groups are normally distributed

    Because gender is between-subjects (not within), we cannot pair individual
    speakers. We therefore treat speaker means as independent observations and
    apply:
      - Shapiro-Wilk on each group
      - Welch t-test if both normal, Wilcoxon rank-sum (Mann-Whitney) otherwise
    This is the standard approach in phonetic studies at the speaker level.
    """
    row = {"phoneme": phoneme, "formant": formant}

    f_vals = spk_means.loc[spk_means["Gender"] == "f", formant].dropna().to_numpy()
    m_vals = spk_means.loc[spk_means["Gender"] == "m", formant].dropna().to_numpy()

    row["n_F"] = len(f_vals)
    row["n_M"] = len(m_vals)
    row["mean_F"] = round(float(np.mean(f_vals)), 4) if len(f_vals) else np.nan
    row["mean_M"] = round(float(np.mean(m_vals)), 4) if len(m_vals) else np.nan
    row["mean_diff_MminusF"] = round(row["mean_M"] - row["mean_F"], 4) \
        if (len(f_vals) and len(m_vals)) else np.nan

    if len(f_vals) < 3 or len(m_vals) < 3:
        row.update({
            "sw_p_F": np.nan, "normal_F": np.nan,
            "sw_p_M": np.nan, "normal_M": np.nan,
            "test_used": "insufficient_data",
            "stat": np.nan, "p_raw": np.nan,
        })
        return row

    # Shapiro-Wilk
    _, sw_p_f = shapiro(f_vals)
    _, sw_p_m = shapiro(m_vals)
    normal_f = bool(sw_p_f > 0.05)
    normal_m = bool(sw_p_m > 0.05)

    row.update({
        "sw_p_F": round(float(sw_p_f), 4),
        "normal_F": normal_f,
        "sw_p_M": round(float(sw_p_m), 4),
        "normal_M": normal_m,
    })

    # Test selection
    # Speaker-level samples are typically small (n ≈ 5–15 per gender),
    # so Wilcoxon rank-sum is preferred unless both groups are clearly normal.
    if normal_f and normal_m:
        from scipy.stats import ttest_ind
        stat, p = ttest_ind(f_vals, m_vals, equal_var=False)  # Welch
        test_name = "welch_t"
    else:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(f_vals, m_vals, alternative="two-sided")
        test_name = "mann_whitney_u"

    row.update({
        "test_used": test_name,
        "stat": round(float(stat), 4),
        "p_raw": round(float(p), 6),
    })
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="./reps/features_acoustic.csv")
    parser.add_argument("--output", default="./tables/test_gender_residual.csv")
    parser.add_argument("--alpha",  type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["phoneme"] = df["phoneme"].fillna("")
    df = lobanov_norm(df)   # Lobanov normalisation applied first

    # Sanity check
    print("Gender values:", df["Gender"].unique())

    vowels = sorted(df[df["phoneme"].apply(is_vowel)]["phoneme"].unique())

    # Aggregate to speaker-level means per phoneme
    # Unit of observation = one speaker × one phoneme → mean F1/F2
    spk_phoneme_means = (
        df[df["phoneme"].apply(is_vowel)]
        .groupby(["spk", "Gender", "phoneme"])[["F1_normed", "F2_normed"]]
        .mean()
        .reset_index()
    )

    rows = []
    for phoneme in vowels:
        sub = spk_phoneme_means[spk_phoneme_means["phoneme"] == phoneme]
        for formant in ("F1_normed", "F2_normed"):
            rows.append(test_gender_one(phoneme, formant, sub))

    results = pd.DataFrame(rows)

    # BH FDR correction
    ran_mask = results["test_used"] != "insufficient_data"
    if ran_mask.sum() > 0:
        p_raw = results.loc[ran_mask, "p_raw"].to_numpy()
        reject, p_adj, _, _ = multipletests(p_raw, alpha=args.alpha, method="fdr_bh")
        results.loc[ran_mask,  "p_adj_BH"]     = np.round(p_adj, 6)
        results.loc[ran_mask,  "reject_H0_BH"] = reject
        results.loc[~ran_mask, "p_adj_BH"]     = np.nan
        results.loc[~ran_mask, "reject_H0_BH"] = np.nan

    col_order = [
        "phoneme", "formant",
        "n_F", "n_M", "mean_F", "mean_M", "mean_diff_MminusF",
        "sw_p_F", "normal_F", "sw_p_M", "normal_M",
        "test_used", "stat", "p_raw", "p_adj_BH", "reject_H0_BH",
    ]
    results = results[col_order].sort_values(["phoneme", "formant"]).reset_index(drop=True)
    results.to_csv(args.output, index=False)

    # Console summary
    sig = results[results["reject_H0_BH"] == True]
    print(f"\nVowels tested : {len(vowels)}")
    print(f"Tests run     : {ran_mask.sum()}")
    print(f"Significant after BH FDR (α={args.alpha}): {len(sig)}\n")
    if len(sig):
        print(sig[["phoneme", "formant", "test_used",
                   "mean_diff_MminusF", "p_raw", "p_adj_BH"]].to_string(index=False))
    else:
        print("No residual gender effect detected after Lobanov normalisation.")
    print(f"\nSaved → {args.output}")

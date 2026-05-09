import argparse
import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from normalise import is_vowel, lobanov_norm


# ── per-vowel test ────────────────────────────────────────────────────────────

def test_one_vowel(phoneme: str, formant: str,
                   g_l1: np.ndarray, g_l2: np.ndarray) -> dict:
    """
    For one (phoneme, formant) pair:
      1. Shapiro-Wilk normality on each group (skip if n < 3 or n > 5000)
      2. Levene's test for equal variances
      3. Two-sample t-test if both groups normal, else Mann-Whitney U
    Returns a dict of all intermediate statistics + the raw p-value.
    """
    row = {"phoneme": phoneme, "formant": formant,
           "n_L1": len(g_l1), "n_L2": len(g_l2)}

    # drop NaNs
    g_l1 = g_l1[~np.isnan(g_l1)]
    g_l2 = g_l2[~np.isnan(g_l2)]
    row["n_L1_valid"] = len(g_l1)
    row["n_L2_valid"] = len(g_l2)

    if len(g_l1) < 3 or len(g_l2) < 3:
        row.update({
            "sw_stat_L1": np.nan, "sw_p_L1": np.nan, "normal_L1": np.nan,
            "sw_stat_L2": np.nan, "sw_p_L2": np.nan, "normal_L2": np.nan,
            "levene_stat": np.nan, "levene_p": np.nan, "equal_var": np.nan,
            "test_used": "insufficient_data",
            "stat": np.nan, "p_raw": np.nan,
            "mean_L1": np.nan, "mean_L2": np.nan, "mean_diff": np.nan,
        })
        return row

    # ── 1. Shapiro-Wilk ──────────────────────────────────────────────────────
    # Shapiro-Wilk is only reliable for n <= 5000; for larger samples
    # it becomes so powerful that trivial deviations are significant.
    def sw(x):
        if len(x) > 5000:
            return np.nan, np.nan   # treat as non-normal if huge sample
        stat, p = shapiro(x)
        return float(stat), float(p)

    sw_stat_l1, sw_p_l1 = sw(g_l1)
    sw_stat_l2, sw_p_l2 = sw(g_l2)
    normal_l1 = bool(sw_p_l1 > 0.05) if not np.isnan(sw_p_l1) else False
    normal_l2 = bool(sw_p_l2 > 0.05) if not np.isnan(sw_p_l2) else False
    both_normal = normal_l1 and normal_l2

    row.update({
        "sw_stat_L1": round(sw_stat_l1, 4) if not np.isnan(sw_stat_l1) else np.nan,
        "sw_p_L1":    round(sw_p_l1,    4) if not np.isnan(sw_p_l1)    else np.nan,
        "normal_L1":  normal_l1,
        "sw_stat_L2": round(sw_stat_l2, 4) if not np.isnan(sw_stat_l2) else np.nan,
        "sw_p_L2":    round(sw_p_l2,    4) if not np.isnan(sw_p_l2)    else np.nan,
        "normal_L2":  normal_l2,
    })

    # ── 2. Levene's test ──────────────────────────────────────────────────────
    lev_stat, lev_p = levene(g_l1, g_l2)
    equal_var = bool(lev_p > 0.05)
    row.update({
        "levene_stat": round(float(lev_stat), 4),
        "levene_p":    round(float(lev_p),    4),
        "equal_var":   equal_var,
    })

    # ── 3. Group comparison ───────────────────────────────────────────────────
    mean_l1 = float(np.mean(g_l1))
    mean_l2 = float(np.mean(g_l2))

    if both_normal:
        # Welch t-test (equal_var=False) when variances differ,
        # Student t-test (equal_var=True) when variances are homogeneous
        stat, p = ttest_ind(g_l1, g_l2, equal_var=equal_var)
        test_name = "student_t" if equal_var else "welch_t"
    else:
        stat, p = mannwhitneyu(g_l1, g_l2, alternative="two-sided")
        test_name = "mann_whitney_u"

    row.update({
        "test_used":  test_name,
        "stat":       round(float(stat), 4),
        "p_raw":      round(float(p),    6),
        "mean_L1":    round(mean_l1, 4),
        "mean_L2":    round(mean_l2, 4),
        "mean_diff":  round(mean_l2 - mean_l1, 4),   # L2 − L1
    })
    return row


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="./reps/features_acoustic.csv")
    parser.add_argument("--output", default="./tables/test_acoustic_l1l2.csv")
    parser.add_argument("--alpha",  type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["phoneme"] = df["phoneme"].fillna("")
    df = lobanov_norm(df)

    vowels = sorted(df[df["phoneme"].apply(is_vowel)]["phoneme"].unique())
    l1_mask = df["L1"] == "fr"
    l2_mask = df["L1"] == "ru"

    rows = []
    for phoneme in vowels:
        ph_mask = df["phoneme"] == phoneme
        for formant in ("F1_normed", "F2_normed"):
            g_l1 = df.loc[ph_mask & l1_mask, formant].to_numpy(dtype=float)
            g_l2 = df.loc[ph_mask & l2_mask, formant].to_numpy(dtype=float)
            rows.append(test_one_vowel(phoneme, formant, g_l1, g_l2))

    results = pd.DataFrame(rows)

    # ── BH FDR correction across all (vowel, formant) pairs ──────────────────
    # Only correct tests that actually ran (not insufficient_data)
    ran_mask = results["test_used"] != "insufficient_data"
    p_raw = results.loc[ran_mask, "p_raw"].to_numpy()

    reject, p_adj, _, _ = multipletests(p_raw, alpha=args.alpha, method="fdr_bh")

    results.loc[ran_mask, "p_adj_BH"]       = np.round(p_adj,   6)
    results.loc[ran_mask, "reject_H0_BH"]   = reject
    results.loc[~ran_mask, "p_adj_BH"]      = np.nan
    results.loc[~ran_mask, "reject_H0_BH"]  = np.nan

    # ── tidy column order ─────────────────────────────────────────────────────
    col_order = [
        "phoneme", "formant",
        "n_L1_valid", "n_L2_valid",
        "mean_L1", "mean_L2", "mean_diff",
        "sw_stat_L1", "sw_p_L1", "normal_L1",
        "sw_stat_L2", "sw_p_L2", "normal_L2",
        "levene_stat", "levene_p", "equal_var",
        "test_used", "stat", "p_raw", "p_adj_BH", "reject_H0_BH",
    ]
    results = results[col_order]
    results = results.sort_values(["phoneme", "formant"]).reset_index(drop=True)

    results.to_csv(args.output, index=False)

    # ── console summary ───────────────────────────────────────────────────────
    sig = results[results["reject_H0_BH"] == True]
    print(f"\nVowels tested: {len(vowels)}  |  Tests run: {ran_mask.sum()}")
    print(f"Significant after BH FDR (α={args.alpha}): {len(sig)}\n")
    print(sig[["phoneme", "formant", "test_used",
               "mean_diff", "p_raw", "p_adj_BH"]].to_string(index=False))
    print(f"\nFull results saved → {args.output}")

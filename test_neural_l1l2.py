import argparse
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from normalise import is_vowel


# ── helpers ───────────────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two mean vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(1.0 - (a @ b) / (na * nb))


def centroid_cosine_dist(reps_l1: np.ndarray, reps_l2: np.ndarray) -> float:
    """Cosine distance between the L1 centroid and the L2 centroid."""
    return cosine_distance(reps_l1.mean(axis=0), reps_l2.mean(axis=0))


def permutation_test(reps_l1: np.ndarray, reps_l2: np.ndarray,
                     n_perm: int, rng: np.random.Generator) -> tuple[float, float]:
    """
    Permutation test on centroid cosine distance.
    Under H0, L1/L2 labels are exchangeable within the pooled set.
    Returns (observed_distance, p_value).
    """
    n1 = len(reps_l1)
    n2 = len(reps_l2)
    pooled = np.concatenate([reps_l1, reps_l2], axis=0)
    n_total = n1 + n2

    obs = centroid_cosine_dist(reps_l1, reps_l2)
    if np.isnan(obs):
        return obs, np.nan

    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n_total)
        null[i] = centroid_cosine_dist(pooled[perm[:n1]], pooled[perm[n1:]])

    # one-sided: observed distance >= null (we test if L1/L2 are further apart
    # than chance; cosine *distance* so larger = more different)
    p = (null >= obs).sum() / n_perm
    # add 1 in numerator and denominator for a conservative estimate
    p = (((null >= obs).sum() + 1) / (n_perm + 1))
    return float(obs), float(p)


# ── per-phoneme test ──────────────────────────────────────────────────────────

def test_one_phoneme(phoneme: str, df_ph: pd.DataFrame,
                     reps: np.ndarray, l1_label: str, l2_label: str,
                     n_perm: int, rng: np.random.Generator) -> dict:

    row = {"phoneme": phoneme}

    idx_l1 = df_ph.index[df_ph["L1"] == l1_label].tolist()
    idx_l2 = df_ph.index[df_ph["L1"] == l2_label].tolist()

    row["n_L1"] = len(idx_l1)
    row["n_L2"] = len(idx_l2)

    if len(idx_l1) < 2 or len(idx_l2) < 2:
        row.update({"obs_cosine_dist": np.nan, "p_raw": np.nan,
                    "test_used": "insufficient_data"})
        return row

    r_l1 = reps[idx_l1]
    r_l2 = reps[idx_l2]

    # drop NaN rows within each group
    r_l1 = r_l1[~np.isnan(r_l1).any(axis=1)]
    r_l2 = r_l2[~np.isnan(r_l2).any(axis=1)]

    if len(r_l1) < 2 or len(r_l2) < 2:
        row.update({"obs_cosine_dist": np.nan, "p_raw": np.nan,
                    "test_used": "insufficient_data"})
        return row

    obs, p = permutation_test(r_l1, r_l2, n_perm=n_perm, rng=rng)
    row.update({
        "obs_cosine_dist": round(obs, 6),
        "p_raw":           round(p,   6),
        "test_used":       "permutation_cosine",
        "n_L1_valid":      len(r_l1),
        "n_L2_valid":      len(r_l2),
    })
    return row


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="./tables/metadata.csv",
                        help="metadata.csv with phoneme, L1, spk columns")
    parser.add_argument("--whisper",  default="./reps/features_whisper.npz")
    parser.add_argument("--xlsr",     default="./reps/features_xlsr.npz")
    parser.add_argument("--n-perm",   type=int, default=5000)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--alpha",    type=float, default=0.05)
    parser.add_argument("--output",   default="./tables/test_neural_l1l2.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.metadata)
    df["phoneme"] = df["phoneme"].fillna("")

    # detect L1/L2 labels automatically
    l1_vals = df["L1"].dropna().unique().tolist()
    print("L1 column unique values:", l1_vals)
    if len(l1_vals) != 2:
        raise ValueError(f"Expected exactly 2 L1 groups, found: {l1_vals}")
    l1_label, l2_label = sorted(l1_vals)   # alphabetical; adjust if needed
    print(f"Treating '{l1_label}' as L1 (native) and '{l2_label}' as L2 (learner)")

    vowels = sorted(df[df["phoneme"].apply(is_vowel)]["phoneme"].unique())

    models = {
        "whisper": args.whisper,
        "xlsr":    args.xlsr,
    }

    all_rows = []

    for model_name, npz_path in models.items():
        print(f"\n── {model_name.upper()} (B={args.n_perm}) ──")
        data = np.load(npz_path, allow_pickle=True)
        reps = data["word_reps"]
        if reps.dtype == object:
            reps = np.stack(reps, axis=0)
        reps = reps.astype(float)

        # reps rows align 1-to-1 with df rows
        assert len(reps) == len(df), \
            f"Row count mismatch: reps={len(reps)}, metadata={len(df)}"

        # work only on vowel rows; keep original positional index for rep slicing
        df_vowels = df[df["phoneme"].apply(is_vowel)].copy()
        # attach reps as a temporary column using positional index
        df_vowels = df_vowels.reset_index(drop=False)   # keeps old index in "index" col
        old_pos = df_vowels["index"].to_numpy()          # positional rows in reps array
        reps_vowels = reps[old_pos]                      # (n_vowel_tokens, d)
        df_vowels = df_vowels.reset_index(drop=True)     # clean 0-based index

        for phoneme in vowels:
            ph_mask = df_vowels["phoneme"] == phoneme
            df_ph = df_vowels[ph_mask].copy()
            reps_ph = reps_vowels[ph_mask.to_numpy()]

            # re-index df_ph from 0 so it aligns with reps_ph rows
            df_ph = df_ph.reset_index(drop=True)

            row = test_one_phoneme(
                phoneme, df_ph, reps_ph,
                l1_label, l2_label,
                n_perm=args.n_perm, rng=rng
            )
            row["model"] = model_name
            all_rows.append(row)
            if row["test_used"] != "insufficient_data":
                print(f"  {phoneme}: d={row['obs_cosine_dist']:.4f}  p={row['p_raw']:.4f}")

    results = pd.DataFrame(all_rows)

    # BH FDR — correct separately per model (two families of tests)
    for model_name in models:
        mask = (results["model"] == model_name) & \
               (results["test_used"] != "insufficient_data")
        if mask.sum() == 0:
            continue
        p_raw = results.loc[mask, "p_raw"].to_numpy()
        reject, p_adj, _, _ = multipletests(p_raw, alpha=args.alpha, method="fdr_bh")
        results.loc[mask,  "p_adj_BH"]     = np.round(p_adj, 6)
        results.loc[mask,  "reject_H0_BH"] = reject
        results.loc[~mask & (results["model"] == model_name),
                    "p_adj_BH"]     = np.nan
        results.loc[~mask & (results["model"] == model_name),
                    "reject_H0_BH"] = np.nan

    col_order = ["model", "phoneme", "n_L1_valid", "n_L2_valid",
                 "obs_cosine_dist", "p_raw", "p_adj_BH", "reject_H0_BH", "test_used"]
    results = results[col_order].sort_values(["model", "phoneme"]).reset_index(drop=True)
    results.to_csv(args.output, index=False)

    # console summary
    print("\n── Summary ──────────────────────────────────────────────────────")
    for model_name in models:
        sub = results[results["model"] == model_name]
        sig = sub[sub["reject_H0_BH"] == True]
        print(f"\n{model_name.upper()}: {len(sig)}/{len(sub)} phonemes significant after BH FDR")
        if len(sig):
            print(sig[["phoneme", "obs_cosine_dist", "p_raw", "p_adj_BH"]].to_string(index=False))

    print(f"\nSaved → {args.output}")

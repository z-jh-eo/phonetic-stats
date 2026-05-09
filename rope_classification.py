import argparse
import numpy as np
import pandas as pd
from normalise import is_vowel


# ── ROPE classification ───────────────────────────────────────────────────────

def classify_rope(ci_lo: float, ci_hi: float,
                  rope_lo: float, rope_hi: float) -> str:
    """
    Equivalent    : CI falls entirely within ROPE
    Non-equivalent: CI falls entirely outside ROPE
    Indeterminate : CI overlaps with ROPE boundary
    """
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return "insufficient_data"
    ci_inside_rope  = (ci_lo >= rope_lo) and (ci_hi <= rope_hi)
    ci_outside_rope = (ci_hi < rope_lo)  or (ci_lo > rope_hi)
    if ci_inside_rope:
        return "equivalent"
    elif ci_outside_rope:
        return "non-equivalent"
    else:
        return "indeterminate"


# ── acoustic ROPE ─────────────────────────────────────────────────────────────
# Spec §8.3.1: JND ≈ 3–5% of formant value ≈ 15–25 Hz for typical F1.
# Default ROPE = [−20, +20] Hz on the *raw* Hz scale.
# Our CIs are in Lobanov units, so we convert: 1 Lobanov unit ≈ 1 SD of F1.
# A typical F1 SD across vowels is ~100 Hz, so 20 Hz ≈ 0.20 Lobanov units.
# We set ROPE_ACOUSTIC = [−0.20, +0.20] and document the conversion.

ROPE_ACOUSTIC_LO = -0.20   # Lobanov units  (≈ −20 Hz)
ROPE_ACOUSTIC_HI =  0.20   # Lobanov units  (≈ +20 Hz)
ROPE_ACOUSTIC_HZ_EQUIV = 20.0   # Hz (for documentation)


# ── neural ROPE ───────────────────────────────────────────────────────────────
# Spec §8.3.2: ROPE = [0, δ₀] where δ₀ = mean intra-speaker cosine distance.
# We compute δ₀ empirically from the data.

def compute_intra_speaker_cosine(df: pd.DataFrame,
                                 reps: np.ndarray,
                                 phoneme: str) -> float:
    """
    Mean cosine distance between pairs of tokens of the same phoneme
    produced by the same speaker. This is the 'noise floor' of the
    representation — the baseline within-speaker variability.
    """
    ph_mask  = (df["phoneme"] == phoneme)
    df_ph    = df[ph_mask].reset_index(drop=True)
    reps_ph  = reps[ph_mask.to_numpy()]

    valid = ~np.isnan(reps_ph).any(axis=1)
    df_ph   = df_ph[valid].reset_index(drop=True)
    reps_ph = reps_ph[valid]

    if len(df_ph) < 2:
        return np.nan

    # normalise rows to unit norm for cosine
    norms = np.linalg.norm(reps_ph, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    Xn    = reps_ph / norms

    dists = []
    for spk, grp in df_ph.groupby("spk"):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            continue
        Xs = Xn[idx]
        sim = Xs @ Xs.T                         # (n_spk, n_spk) cosine sim
        # upper triangle excluding diagonal → cosine distances
        ut = np.triu_indices(len(idx), k=1)
        dists.extend((1.0 - sim[ut]).tolist())

    return float(np.mean(dists)) if dists else np.nan


def compute_delta0(df: pd.DataFrame, reps: np.ndarray,
                   vowels: list) -> float:
    """δ₀ = mean intra-speaker cosine distance averaged across all vowels."""
    per_vowel = [compute_intra_speaker_cosine(df, reps, ph) for ph in vowels]
    valid = [x for x in per_vowel if not np.isnan(x)]
    return float(np.mean(valid)) if valid else np.nan


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-ci",  default="./tables/lme_ci_acoustic.csv",
                        help="Output of lme_ci_forest.py")
    parser.add_argument("--neural-ci",    default="./tables/neural_ci_cosine.csv",
                        help="Output of neural_ci_forest.py")
    parser.add_argument("--metadata",     default="./tables/metadata.csv")
    parser.add_argument("--whisper",      default="./reps/features_whisper.npz")
    parser.add_argument("--xlsr",         default="./reps/features_xlsr.npz")
    parser.add_argument("--rope-ac-lo",   type=float, default=ROPE_ACOUSTIC_LO)
    parser.add_argument("--rope-ac-hi",   type=float, default=ROPE_ACOUSTIC_HI)
    parser.add_argument("--out-table",    default="./tables/rope_classification.csv")
    parser.add_argument("--out-summary",  default="./tables/rope_summary.csv")
    args = parser.parse_args()

    # ── load acoustic CIs ─────────────────────────────────────────────────────
    ac_ci = pd.read_csv(args.acoustic_ci)
    # keep only L1 main effect and L1:Gender interaction for F1 and F2
    ac_ci = ac_ci[ac_ci["effect"].str.contains("L1", case=False)].copy()

    # ── load neural CIs ───────────────────────────────────────────────────────
    ne_ci = pd.read_csv(args.neural_ci)
    ne_ci = ne_ci[ne_ci["status"] == "ok"].copy()

    # ── compute neural ROPE threshold δ₀ ─────────────────────────────────────
    df_meta = pd.read_csv(args.metadata)
    df_meta["phoneme"] = df_meta["phoneme"].fillna("")
    vowels     = sorted(df_meta[df_meta["phoneme"].apply(is_vowel)]["phoneme"].unique())
    vowel_pos  = df_meta[df_meta["phoneme"].apply(is_vowel)].index.to_numpy()
    df_v       = df_meta[df_meta["phoneme"].apply(is_vowel)].copy().reset_index(drop=True)

    def load_reps(path):
        data = np.load(path, allow_pickle=True)
        reps = data["word_reps"]
        if reps.dtype == object:
            reps = np.stack(reps, axis=0)
        return reps.astype(float)[vowel_pos]

    print("Computing intra-speaker cosine distances for neural ROPE …")
    delta0 = {}
    for model_name, path in [("whisper", args.whisper), ("xlsr", args.xlsr)]:
        reps = load_reps(path)
        d0   = compute_delta0(df_v, reps, vowels)
        delta0[model_name] = d0
        print(f"  δ₀ ({model_name}) = {d0:.6f}  →  ROPE = [0, {d0:.6f}]")

    # ── classify acoustic contrasts ───────────────────────────────────────────
    rows = []

    for _, row in ac_ci.iterrows():
        classification = classify_rope(
            row["ci_lo"], row["ci_hi"],
            args.rope_ac_lo, args.rope_ac_hi
        )
        rows.append({
            "representation": f"acoustic_{row['formant']}",
            "phoneme":        row["phoneme"],
            "effect":         row["effect"],
            "point_est":      round(row["coef"],   4),
            "ci_lo":          round(row["ci_lo"],  4),
            "ci_hi":          round(row["ci_hi"],  4),
            "rope_lo":        round(args.rope_ac_lo, 4),
            "rope_hi":        round(args.rope_ac_hi, 4),
            "rope_note":      f"±{ROPE_ACOUSTIC_HZ_EQUIV:.0f} Hz ≈ ±{abs(args.rope_ac_lo):.2f} Lobanov",
            "classification": classification,
            "p_value":        round(row["p_value"], 6)
                              if "p_value" in row and not pd.isna(row["p_value"])
                              else np.nan,
        })

    # ── classify neural contrasts ─────────────────────────────────────────────
    for _, row in ne_ci.iterrows():
        model = row["model"]
        d0    = delta0.get(model, np.nan)
        rope_lo = 0.0
        rope_hi = d0 if not np.isnan(d0) else np.nan

        if np.isnan(rope_hi):
            classification = "rope_undefined"
        else:
            classification = classify_rope(
                row["ci_lo"], row["ci_hi"], rope_lo, rope_hi
            )

        rows.append({
            "representation": f"neural_{model}",
            "phoneme":        row["phoneme"],
            "effect":         "L1_vs_L2_centroid_distance",
            "point_est":      round(row["obs_dist"], 6),
            "ci_lo":          round(row["ci_lo"],    6),
            "ci_hi":          round(row["ci_hi"],    6),
            "rope_lo":        round(rope_lo, 6),
            "rope_hi":        round(rope_hi, 6) if not np.isnan(rope_hi) else np.nan,
            "rope_note":      f"[0, δ₀={rope_hi:.4f}] mean intra-spk cosine dist"
                              if not np.isnan(rope_hi) else "δ₀ undefined",
            "classification": classification,
            "p_value":        np.nan,   # from permutation test, not LME
        })

    results = pd.DataFrame(rows)
    results.to_csv(args.out_table, index=False)
    print(f"\nSaved → {args.out_table}")

    # ── summary table ─────────────────────────────────────────────────────────
    summary = (
        results.groupby(["representation", "classification"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # ensure all classification columns present
    for col in ["equivalent", "non-equivalent", "indeterminate",
                "insufficient_data", "rope_undefined"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["total"] = summary[
        [c for c in summary.columns if c != "representation"]
    ].sum(axis=1)
    summary["pct_non_equivalent"] = (
        summary.get("non-equivalent", 0) / summary["total"] * 100
    ).round(1)

    summary.to_csv(args.out_summary, index=False)
    print(f"Saved → {args.out_summary}")

    # ── console report ────────────────────────────────────────────────────────
    print(f"\n── ROPE parameters ──────────────────────────────────────────────")
    print(f"  Acoustic : [{args.rope_ac_lo}, {args.rope_ac_hi}] Lobanov units"
          f"  (≈ ±{ROPE_ACOUSTIC_HZ_EQUIV:.0f} Hz)")
    for model_name, d0 in delta0.items():
        print(f"  Neural ({model_name}): [0, {d0:.4f}]  (mean intra-speaker cosine dist)")

    print(f"\n── Classification counts ─────────────────────────────────────────")
    print(summary.to_string(index=False))

    print(f"\n── Non-equivalent contrasts ──────────────────────────────────────")
    ne = results[results["classification"] == "non-equivalent"]
    if len(ne):
        print(ne[["representation", "phoneme", "effect",
                  "point_est", "ci_lo", "ci_hi",
                  "rope_lo", "rope_hi"]].to_string(index=False))
    else:
        print("  None found.")

    print(f"\n── Statistically significant but equivalent (acoustic) ───────────")
    sig_equiv = results[
        (results["representation"].str.startswith("acoustic")) &
        (results["classification"] == "equivalent") &
        (results["p_value"] < 0.05)
    ]
    if len(sig_equiv):
        print(sig_equiv[["representation", "phoneme", "effect",
                          "point_est", "ci_lo", "ci_hi", "p_value"]].to_string(index=False))
    else:
        print("  None found.")

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from normalise import is_vowel


# ── cosine distance between two centroid vectors ──────────────────────────────

def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(1.0 - (a @ b) / (na * nb))


# ── speaker-level bootstrap CI ────────────────────────────────────────────────

def bootstrap_ci(df: pd.DataFrame, reps: np.ndarray,
                 phoneme: str, l1_label: str, l2_label: str,
                 n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42) -> dict:
    """
    95% bootstrap CI on cosine distance between L1 and L2 centroids
    for a given phoneme. Resampling at the speaker level (spec §10.2):
    draw l speakers with replacement, use all their tokens.
    """
    rng = np.random.default_rng(seed)
    speakers = df["spk"].unique()
    n_spk    = len(speakers)

    ph_mask = df["phoneme"] == phoneme
    df_ph   = df[ph_mask].reset_index(drop=True)
    reps_ph = reps[ph_mask.to_numpy()]

    # remove NaN rep rows
    valid = ~np.isnan(reps_ph).any(axis=1)
    df_ph   = df_ph[valid].reset_index(drop=True)
    reps_ph = reps_ph[valid]

    n_l1 = (df_ph["L1"] == l1_label).sum()
    n_l2 = (df_ph["L1"] == l2_label).sum()

    if n_l1 < 2 or n_l2 < 2:
        return {"phoneme": phoneme, "n_L1": int(n_l1), "n_L2": int(n_l2),
                "obs_dist": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "status": "insufficient"}

    # observed distance (full sample)
    c_l1 = reps_ph[df_ph["L1"] == l1_label].mean(axis=0)
    c_l2 = reps_ph[df_ph["L1"] == l2_label].mean(axis=0)
    obs  = cosine_dist(c_l1, c_l2)

    boot = np.empty(n_boot)
    for b in range(n_boot):
        # resample speakers with replacement
        sampled = rng.choice(speakers, size=n_spk, replace=True)
        idx = np.concatenate([
            df_ph.index[df_ph["spk"] == s].to_numpy() for s in sampled
        ])
        if len(idx) == 0:
            boot[b] = np.nan
            continue
        df_b   = df_ph.loc[idx]
        reps_b = reps_ph[idx]

        r_l1 = reps_b[df_b["L1"].to_numpy() == l1_label]
        r_l2 = reps_b[df_b["L1"].to_numpy() == l2_label]
        if len(r_l1) < 1 or len(r_l2) < 1:
            boot[b] = np.nan
            continue
        boot[b] = cosine_dist(r_l1.mean(axis=0), r_l2.mean(axis=0))

    lo = float(np.nanquantile(boot, alpha / 2))
    hi = float(np.nanquantile(boot, 1 - alpha / 2))

    return {
        "phoneme":  phoneme,
        "n_L1":     int(n_l1),
        "n_L2":     int(n_l2),
        "obs_dist": round(obs, 6),
        "ci_lo":    round(lo,  6),
        "ci_hi":    round(hi,  6),
        "status":   "ok",
    }


# ── forest plot ───────────────────────────────────────────────────────────────

def forest_plot(results: dict[str, pd.DataFrame],
                outpath: str, alpha: float = 0.05) -> None:
    """
    One panel per model (Whisper / XLS-R).
    Phonemes on y-axis, cosine distance + 95% CI on x-axis.
    Red = CI entirely above 0 (clearly non-zero distance).
    Blue = CI includes 0.
    """
    model_names = list(results.keys())
    n_panels    = len(model_names)
    all_phonemes = sorted(set().union(*[set(df["phoneme"]) for df in results.values()]))

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5.5 * n_panels, max(4, len(all_phonemes) * 0.6)),
        sharey=True
    )
    if n_panels == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        df = results[model_name].copy()
        df = df.set_index("phoneme").reindex(all_phonemes).reset_index()

        y_pos   = np.arange(len(all_phonemes))
        obs     = df["obs_dist"].to_numpy(dtype=float)
        ci_lo   = df["ci_lo"].to_numpy(dtype=float)
        ci_hi   = df["ci_hi"].to_numpy(dtype=float)

        # significant = CI entirely above 0 (distance > 0 with 95% confidence)
        sig = ci_lo > 0

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        for i, (y, d, lo, hi, s) in enumerate(zip(y_pos, obs, ci_lo, ci_hi, sig)):
            if np.isnan(d):
                continue
            col = "#c0392b" if s else "#2980b9"
            ax.plot([lo, hi], [y, y], color=col, linewidth=1.8)
            ax.plot(d, y, "o", color=col, markersize=5, zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_phonemes, fontsize=10)
        ax.set_xlabel("Cosine distance (L1 vs L2 centroid)", fontsize=9)
        ax.set_title(model_name.upper(), fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(left=min(-0.01,
                             np.nanmin(ci_lo) - 0.005 if not np.all(np.isnan(ci_lo)) else -0.01))

    sig_patch   = mpatches.Patch(color="#c0392b", label="CI > 0 (non-zero distance)")
    nosig_patch = mpatches.Patch(color="#2980b9", label="CI includes 0")
    fig.legend(handles=[sig_patch, nosig_patch],
               loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Bootstrap CIs on L1 vs L2 cosine distance\n"
                 f"(B=2000, speaker-level resampling, α={alpha})",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="./tables/metadata.csv")
    parser.add_argument("--whisper",  default="./reps/features_whisper.npz")
    parser.add_argument("--xlsr",     default="./reps/features_xlsr.npz")
    parser.add_argument("--n-boot",   type=int,   default=2000)
    parser.add_argument("--alpha",    type=float, default=0.05)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--out-table",default="./tables/neural_ci_cosine.csv")
    parser.add_argument("--out-plot", default="./plots/forest_neural_cosine.png")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out_table), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_plot),  exist_ok=True)

    df = pd.read_csv(args.metadata)
    df["phoneme"] = df["phoneme"].fillna("")

    # detect L1/L2 labels
    l1_vals = sorted(df["L1"].dropna().unique().tolist())
    if len(l1_vals) != 2:
        raise ValueError(f"Expected 2 L1 groups, got: {l1_vals}")
    l1_label, l2_label = l1_vals
    print(f"L1='{l1_label}'  L2='{l2_label}'")

    vowels = sorted(df[df["phoneme"].apply(is_vowel)]["phoneme"].unique())
    print(f"Vowels: {vowels}")

    # filter to vowel rows; keep positional index for rep slicing
    vowel_pos = df[df["phoneme"].apply(is_vowel)].index.to_numpy()
    df_v = df[df["phoneme"].apply(is_vowel)].copy().reset_index(drop=True)

    def load_reps(path: str) -> np.ndarray:
        data = np.load(path, allow_pickle=True)
        reps = data["word_reps"]
        if reps.dtype == object:
            reps = np.stack(reps, axis=0)
        return reps.astype(float)[vowel_pos]

    models = {"whisper": args.whisper, "xlsr": args.xlsr}
    all_rows  = []
    plot_data = {}

    for model_name, npz_path in models.items():
        print(f"\n── {model_name.upper()} (B={args.n_boot}) ──────────────────")
        reps = load_reps(npz_path)

        rows = []
        for phoneme in vowels:
            res = bootstrap_ci(
                df_v, reps, phoneme, l1_label, l2_label,
                n_boot=args.n_boot, alpha=args.alpha, seed=args.seed
            )
            res["model"] = model_name
            rows.append(res)
            if res["status"] == "ok":
                print(f"  {phoneme}: d={res['obs_dist']:.4f} "
                      f"[{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]")
            else:
                print(f"  {phoneme}: {res['status']}")

        plot_data[model_name] = pd.DataFrame(rows)
        all_rows.extend(rows)

    # save table
    col_order = ["model", "phoneme", "n_L1", "n_L2",
                 "obs_dist", "ci_lo", "ci_hi", "status"]
    out_df = pd.DataFrame(all_rows)[col_order]
    out_df.to_csv(args.out_table, index=False)
    print(f"\nSaved → {args.out_table}")

    # forest plot
    print("Generating forest plot …")
    forest_plot(plot_data, args.out_plot, alpha=args.alpha)

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
from normalise import is_vowel, lobanov_norm

warnings.filterwarnings("ignore")


VOWEL_HEIGHT = {
    "i": "high", "y": "high", "ɪ":"high", "u": "high",
    "e": "mid",  "ø": "mid",  "o": "mid",
    "ɛ": "mid",  "œ": "mid",
    "a": "low",  "ɑ": "low",
}


# ── fit M1 main-effects model and extract profile CIs ────────────────────────

def fit_and_ci(sub: pd.DataFrame, formant: str,
               alpha: float = 0.05) -> dict | None:
    """
    Fit F ~ L1 + Gender with random speaker intercept (ML).
    Return profile-likelihood CIs for L1 and Gender fixed effects,
    plus the L1×Gender interaction contrast from a separate M2 fit.
    Falls back to Wald CIs if profile CI fails (can be slow/unstable).
    """
    sub = sub.dropna(subset=[formant, "L1", "Gender", "spk"]).copy()
    sub["L1"]     = sub["L1"].astype("category")
    sub["Gender"] = sub["Gender"].astype("category")

    if sub["spk"].nunique() < 3 or len(sub) < 10:
        return None

    # ── M1: main effects (ML) ────────────────────────────────────────────────
    try:
        md  = smf.mixedlm(f"{formant} ~ L1 + Gender", sub,
                          groups=sub["spk"])
        mdf = md.fit(reml=False, method=["lbfgs", "bfgs", "cg"],
                     maxiter=1000)
    except Exception:
        return None

    results = {}

    # collect all fixed-effect parameter names (excluding Intercept and Group Var)
    fe_params = [p for p in mdf.params.index
                 if p not in ("Intercept", "Group Var")]

    for param in fe_params:
        coef = float(mdf.params[param])
        se   = float(mdf.bse[param])
        pval = float(mdf.pvalues[param])

        # profile-likelihood CI — try first, fall back to Wald
        try:
            ci_df = mdf.conf_int(alpha=alpha)
            ci_lo = float(ci_df.loc[param, 0])
            ci_hi = float(ci_df.loc[param, 1])
            ci_method = "wald"   # statsmodels mixedlm uses Wald by default
        except Exception:
            z = 1.959964  # 97.5th percentile
            ci_lo = coef - z * se
            ci_hi = coef + z * se
            ci_method = "wald_fallback"

        results[param] = {
            "coef": round(coef, 4),
            "se":   round(se,   4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "p_value": round(pval, 6),
            "ci_method": ci_method,
        }

    # ── M2: interaction contrast L1 × Gender ────────────────────────────────
    try:
        md2  = smf.mixedlm(f"{formant} ~ L1 * Gender", sub,
                           groups=sub["spk"])
        mdf2 = md2.fit(reml=False, method=["lbfgs", "bfgs", "cg"],
                       maxiter=1000)
        ix_params = [p for p in mdf2.params.index
                     if ":" in p]   # interaction terms have ":" in patsy
        for param in ix_params:
            coef = float(mdf2.params[param])
            se   = float(mdf2.bse[param])
            pval = float(mdf2.pvalues[param])
            try:
                ci_df = mdf2.conf_int(alpha=alpha)
                ci_lo = float(ci_df.loc[param, 0])
                ci_hi = float(ci_df.loc[param, 1])
            except Exception:
                z = 1.959964
                ci_lo = coef - z * se
                ci_hi = coef + z * se
            results[param] = {
                "coef": round(coef, 4),
                "se":   round(se,   4),
                "ci_lo": round(ci_lo, 4),
                "ci_hi": round(ci_hi, 4),
                "p_value": round(pval, 6),
                "ci_method": "wald_interaction",
            }
    except Exception:
        pass

    return results


# ── forest plot ───────────────────────────────────────────────────────────────

def forest_plot(df: pd.DataFrame, formant: str,
                outpath: str, alpha: float = 0.05) -> None:
    """
    One panel per effect (L1, Gender, interaction).
    Phonemes on y-axis, point estimate + CI on x-axis.
    Significant contrasts (CI excludes zero) shown in a distinct colour.
    """
    effects = df["effect"].unique()
    n_effects = len(effects)

    fig, axes = plt.subplots(1, n_effects,
                             figsize=(5 * n_effects, max(4, len(df["phoneme"].unique()) * 0.55)),
                             sharey=False)
    if n_effects == 1:
        axes = [axes]

    vowels = sorted(df["phoneme"].unique())

    for ax, eff in zip(axes, effects):
        sub = df[df["effect"] == eff].copy()
        sub = sub.set_index("phoneme").reindex(vowels).reset_index()

        y_pos  = np.arange(len(vowels))
        coefs  = sub["coef"].to_numpy(dtype=float)
        ci_lo  = sub["ci_lo"].to_numpy(dtype=float)
        ci_hi  = sub["ci_hi"].to_numpy(dtype=float)
        pvals  = sub["p_value"].to_numpy(dtype=float)

        sig = (ci_lo > 0) | (ci_hi < 0)   # CI excludes zero

        colors = ["#c0392b" if s else "#2980b9" for s in sig]

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        for i, (y, c, lo, hi, col) in enumerate(zip(y_pos, coefs, ci_lo, ci_hi, colors)):
            ax.plot([lo, hi], [y, y], color=col, linewidth=1.5)
            ax.plot(c, y, "o", color=col, markersize=5, zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(vowels, fontsize=10)
        ax.set_xlabel("Estimate (Lobanov units)", fontsize=9)
        ax.set_title(f"{eff}\n({formant})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    sig_patch   = mpatches.Patch(color="#c0392b", label="CI excludes zero")
    nosig_patch = mpatches.Patch(color="#2980b9", label="CI includes zero")
    fig.legend(handles=[sig_patch, nosig_patch],
               loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f"Fixed-effect CIs — {formant}", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic",   default="./reps/features_acoustic.csv")
    parser.add_argument("--alpha",      type=float, default=0.05)
    parser.add_argument("--out-table",  default="./tables/lme_ci_acoustic.csv")
    parser.add_argument("--out-plot-f1",default="./plots/forest_F1.png")
    parser.add_argument("--out-plot-f2",default="./plots/forest_F2.png")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out_table),   exist_ok=True)
    os.makedirs(os.path.dirname(args.out_plot_f1), exist_ok=True)

    df = pd.read_csv(args.acoustic)
    df["phoneme"] = df["phoneme"].fillna("")
    df = lobanov_norm(df)
    df_v = df[df["phoneme"].apply(is_vowel)].copy().reset_index(drop=True)

    vowels = sorted(df_v["phoneme"].unique())
    print(f"Vowels : {vowels}")
    print(f"Speakers: {df_v['spk'].nunique()}  |  Tokens: {len(df_v)}\n")

    all_rows = []

    for formant in ("F1_normed", "F2_normed"):
        print(f"── {formant} ──────────────────────────────────────────────")
        for phoneme in vowels:
            sub = df_v[df_v["phoneme"] == phoneme]
            res = fit_and_ci(sub, formant, alpha=args.alpha)
            if res is None:
                print(f"  {phoneme}: skipped")
                continue
            for effect, vals in res.items():
                row = {"phoneme": phoneme, "formant": formant,
                       "effect": effect, **vals}
                all_rows.append(row)
            effects_str = {k: f"{v['coef']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]"
                           for k, v in res.items()}
            print(f"  {phoneme}: " + "  |  ".join(
                f"{k}={v}" for k, v in effects_str.items()))

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(args.out_table, index=False)
    print(f"\nSaved → {args.out_table}")

    # ── forest plots ──────────────────────────────────────────────────────────
    print("\nGenerating forest plots …")
    for formant, outpath in [("F1_normed", args.out_plot_f1),
                              ("F2_normed", args.out_plot_f2)]:
        sub = results_df[results_df["formant"] == formant]
        if sub.empty:
            continue
        forest_plot(sub, formant, outpath, alpha=args.alpha)

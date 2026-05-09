import argparse
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from normalise import is_vowel, lobanov_norm

warnings.filterwarnings("ignore")


# ── vowel height coding ───────────────────────────────────────────────────────
# Standard French vowel height classification
VOWEL_HEIGHT = {
    "i": "high", "y": "high", "ɪ":"high", "u": "high",
    "e": "mid",  "ø": "mid",  "o": "mid",
    "ɛ": "mid",  "œ": "mid",
    "a": "low",  "ɑ": "low",
}


def add_vowel_height(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vowel_height"] = df["phoneme"].map(VOWEL_HEIGHT).fillna("unknown")
    return df


# ── ICC from null model ───────────────────────────────────────────────────────

def compute_icc(mdf) -> float:
    var_u = float(mdf.cov_re.iloc[0, 0])
    var_e = float(mdf.scale)
    return var_u / (var_u + var_e) if (var_u + var_e) > 0 else np.nan


# ── marginal and conditional R² (Nakagawa & Schielzeth 2013) ─────────────────

def r2_lme(mdf) -> tuple[float, float]:
    """
    Marginal R²  = variance explained by fixed effects only.
    Conditional R² = variance explained by fixed + random effects.
 
    Uses the variance-components approach (Nakagawa & Schielzeth 2013).
    Handles the boundary case where var_random collapses to 0 (singular
    cov_re), which makes mdf.fittedvalues and mdf.random_effects crash.
    In that case var_fixed is computed directly from Xβ̂ = exog @ params_fe.
    """
    var_random = float(mdf.cov_re.iloc[0, 0])
    var_resid  = float(mdf.scale)
 
    # singular boundary solution: random variance == 0
    # fittedvalues / random_effects will crash, so compute Xβ̂ directly
    if var_random < 1e-10:
        fe_params = mdf.fe_params.to_numpy()          # fixed-effect coefficients
        Xb = mdf.model.exog @ fe_params               # (n,) fixed predictions
        var_fixed = float(np.var(Xb, ddof=1))
    else:
        try:
            # subtract per-group random intercept from total fitted values
            re_vals = pd.Series(0.0, index=mdf.model.endog_names
                                if hasattr(mdf.model, "endog_names")
                                else range(len(mdf.model.endog)))
            re_vals = pd.Series(0.0, index=range(len(mdf.model.endog)))
            groups  = mdf.model.groups.to_numpy() \
                      if hasattr(mdf.model.groups, "to_numpy") \
                      else np.array(mdf.model.groups)
            for grp, re_df in mdf.random_effects.items():
                mask = groups == grp
                re_vals[mask] = float(re_df.iloc[0])
            fe_params = mdf.fe_params.to_numpy()
            Xb = mdf.model.exog @ fe_params
            var_fixed = float(np.var(Xb, ddof=1))
        except Exception:
            fe_params = mdf.fe_params.to_numpy()
            Xb = mdf.model.exog @ fe_params
            var_fixed = float(np.var(Xb, ddof=1))
 
    var_total      = var_fixed + var_random + var_resid
    r2_marginal    = var_fixed / var_total if var_total > 0 else np.nan
    r2_conditional = (var_fixed + var_random) / var_total if var_total > 0 else np.nan
    return round(r2_marginal, 4), round(r2_conditional, 4)


# ── fit one model safely ──────────────────────────────────────────────────────

def fit_model(formula: str, data: pd.DataFrame,
              groups: str = "spk",
              re_formula: str = None,
              reml: bool = True) -> tuple:
    """
    Returns (result, converged). reml=False for LRT comparisons.
    """
    try:
        md = smf.mixedlm(formula, data, groups=data[groups],
                         re_formula=re_formula)
        mdf = md.fit(reml=reml, method=["lbfgs", "bfgs", "cg"],
                     maxiter=1000)
        return mdf, True
    except Exception as e:
        return None, False


# ── likelihood-ratio test ─────────────────────────────────────────────────────

def lrt(mdf_null, mdf_full) -> dict:
    """LRT between nested models (both fitted with REML=False / ML)."""
    from scipy.stats import chi2 as chi2_dist
    ll_null = mdf_null.llf
    ll_full = mdf_full.llf
    stat = 2 * (ll_full - ll_null)
    df_diff = mdf_full.df_modelwc - mdf_null.df_modelwc
    df_diff = max(df_diff, 1)
    p = float(chi2_dist.sf(stat, df_diff))
    return {"LRT_stat": round(float(stat), 4),
            "df_diff": int(df_diff),
            "p_LRT": round(p, 6)}


# ── model summary row ─────────────────────────────────────────────────────────

def summary_row(name: str, mdf, phoneme: str,
                formant: str, reml: bool = True) -> dict:
    icc = compute_icc(mdf)
    r2m, r2c = r2_lme(mdf)
    row = {
        "phoneme":    phoneme,
        "formant":    formant,
        "model":      name,
        "AIC":        round(float(mdf.aic), 2),
        "BIC":        round(float(mdf.bic), 2),
        "loglik":     round(float(mdf.llf), 4),
        "ICC":        round(icc, 4),
        "R2_marginal":    r2m,
        "R2_conditional": r2c,
        "REML":       reml,
    }
    # fixed-effect coefficients
    for param, coef in mdf.params.items():
        row[f"coef_{param}"] = round(float(coef), 4)
    for param, pval in mdf.pvalues.items():
        row[f"pval_{param}"] = round(float(pval), 6)
    return row


# ── acoustic LME pipeline ─────────────────────────────────────────────────────

def run_acoustic_lme(df: pd.DataFrame, phoneme: str,
                     formant: str) -> tuple[list[dict], list[dict]]:
    """
    Five-step model building for one (phoneme, formant) combination.
    Returns (summary_rows, lrt_rows).
    """
    sub = df[df["phoneme"] == phoneme].dropna(
        subset=[formant, "L1", "Gender", "vowel_height", "spk"]
    ).copy()

    if sub["spk"].nunique() < 3 or len(sub) < 10:
        return [], []

    # ensure categorical coding
    sub["L1"]          = sub["L1"].astype("category")
    sub["Gender"]      = sub["Gender"].astype("category")
    sub["vowel_height"] = sub["vowel_height"].astype("category")

    rows = []
    lrt_rows = []

    # ── M0: null (random intercept only) — REML for ICC ──────────────────────
    m0_reml, ok = fit_model(f"{formant} ~ 1", sub, reml=True)
    if not ok:
        return [], []
    rows.append(summary_row("M0_null", m0_reml, phoneme, formant, reml=True))

    # refit M0 with ML for LRT
    m0_ml, _ = fit_model(f"{formant} ~ 1", sub, reml=False)

    # ── M1: main effects — ML ─────────────────────────────────────────────────
    m1_ml, ok = fit_model(f"{formant} ~ L1 + Gender", sub, reml=False)
    if not ok:
        return rows, lrt_rows
    rows.append(summary_row("M1_main", m1_ml, phoneme, formant, reml=False))
    if m0_ml:
        lr = lrt(m0_ml, m1_ml)
        lr.update({"phoneme": phoneme, "formant": formant,
                   "comparison": "M0_null vs M1_main"})
        lrt_rows.append(lr)

    # ── M2: full (add L1 × Gender interaction) — ML ───────────────────────────
    m2_ml, ok = fit_model(f"{formant} ~ L1 * Gender", sub, reml=False)
    if not ok:
        return rows, lrt_rows
    rows.append(summary_row("M2_interaction", m2_ml, phoneme, formant, reml=False))
    lr = lrt(m1_ml, m2_ml)
    lr.update({"phoneme": phoneme, "formant": formant,
               "comparison": "M1_main vs M2_interaction"})
    lrt_rows.append(lr)

    # ── M3: add vowel height context — ML ────────────────────────────────────
    # vowel_height is constant within phoneme, so this only varies across
    # phonemes in a pooled model. For a single-phoneme model it's collinear;
    # include it only if more than one height level is present (pooled runs).
    if sub["vowel_height"].nunique() > 1:
        m3_ml, ok = fit_model(
            f"{formant} ~ L1 * Gender + vowel_height", sub, reml=False)
        if ok:
            rows.append(summary_row("M3_context", m3_ml,
                                    phoneme, formant, reml=False))
            lr = lrt(m2_ml, m3_ml)
            lr.update({"phoneme": phoneme, "formant": formant,
                       "comparison": "M2_interaction vs M3_context"})
            lrt_rows.append(lr)
            best_ml = m3_ml
        else:
            best_ml = m2_ml
    else:
        best_ml = m2_ml

    # ── M4: random slope for L1 by speaker — ML ──────────────────────────────
    # re_formula="~L1" adds a random slope for L1 within each speaker
    m4_ml, ok = fit_model(
        f"{formant} ~ L1 * Gender", sub,
        re_formula="~L1", reml=False)
    if ok:
        rows.append(summary_row("M4_random_slope", m4_ml,
                                phoneme, formant, reml=False))
        lr = lrt(best_ml, m4_ml)
        lr.update({"phoneme": phoneme, "formant": formant,
                   "comparison": "M2/M3 vs M4_random_slope"})
        lrt_rows.append(lr)

    return rows, lrt_rows


# ── neural LME pipeline ───────────────────────────────────────────────────────

def run_neural_lme(df: pd.DataFrame, reps: np.ndarray,
                   model_name: str, phoneme: str,
                   n_pc: int = 5) -> list[dict]:
    """
    Project reps to d=n_pc PCs (fit on training tokens for this phoneme),
    then fit M1 (main effects) LME for each PC dimension.
    Returns summary rows with marginal R² per PC.
    """
    ph_mask = df["phoneme"] == phoneme
    sub_df  = df[ph_mask].copy().reset_index(drop=True)
    sub_rep = reps[ph_mask.to_numpy()]

    # drop NaN rows
    nan_mask = np.isnan(sub_rep).any(axis=1)
    sub_df  = sub_df[~nan_mask].reset_index(drop=True)
    sub_rep = sub_rep[~nan_mask]

    if sub_df["spk"].nunique() < 3 or len(sub_df) < 10:
        return []

    # PCA on this phoneme's tokens
    n_pc_actual = min(n_pc, sub_rep.shape[1], len(sub_rep) - 1)
    pca = PCA(n_components=n_pc_actual)
    pcs = pca.fit_transform(sub_rep)   # (n_tokens, n_pc_actual)

    sub_df["L1"]     = sub_df["L1"].astype("category")
    sub_df["Gender"] = sub_df["Gender"].astype("category")

    rows = []
    for k in range(n_pc_actual):
        col = f"PC{k+1}"
        sub_df[col] = pcs[:, k]
        mdf, ok = fit_model(f"{col} ~ L1 + Gender", sub_df, reml=False)
        if not ok:
            continue
        r2m, r2c = r2_lme(mdf)
        icc = compute_icc(mdf)
        row = {
            "neural_model": model_name,
            "phoneme":  phoneme,
            "PC":       col,
            "pct_var_explained": round(float(pca.explained_variance_ratio_[k] * 100), 2),
            "AIC":      round(float(mdf.aic), 2),
            "BIC":      round(float(mdf.bic), 2),
            "ICC":      round(icc, 4),
            "R2_marginal":    r2m,
            "R2_conditional": r2c,
        }
        for param, coef in mdf.params.items():
            row[f"coef_{param}"] = round(float(coef), 4)
        for param, pval in mdf.pvalues.items():
            row[f"pval_{param}"] = round(float(pval), 6)
        rows.append(row)

    return rows


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic",  default="./reps/features_acoustic.csv")
    parser.add_argument("--whisper",   default="./reps/features_whisper.npz")
    parser.add_argument("--xlsr",      default="./reps/features_xlsr.npz")
    parser.add_argument("--n-pc",      type=int, default=5,
                        help="Number of PCs for neural LME")
    parser.add_argument("--out-acoustic-models",
                        default="./tables/lme_acoustic_models.csv")
    parser.add_argument("--out-acoustic-lrt",
                        default="./tables/lme_acoustic_lrt.csv")
    parser.add_argument("--out-neural",
                        default="./tables/lme_neural_models.csv")
    args = parser.parse_args()

    # ── load & prepare ────────────────────────────────────────────────────────
    df = pd.read_csv(args.acoustic)
    df["phoneme"] = df["phoneme"].fillna("")
    df = lobanov_norm(df)
    df = add_vowel_height(df)
    df_v = df[df["phoneme"].apply(is_vowel)].copy().reset_index(drop=True)
    vowel_pos = df[df["phoneme"].apply(is_vowel)].index.to_numpy()

    vowels = sorted(df_v["phoneme"].unique())
    print(f"Vowels: {vowels}")
    print(f"Speakers: {df_v['spk'].nunique()}  |  Tokens: {len(df_v)}\n")

    # ── acoustic LME ─────────────────────────────────────────────────────────
    print("── Acoustic LME ─────────────────────────────────────────────────")
    ac_rows, lrt_rows_all = [], []

    for phoneme in vowels:
        for formant in ("F1_normed", "F2_normed"):
            print(f"  {phoneme} / {formant} …", end=" ")
            r, l = run_acoustic_lme(df_v, phoneme, formant)
            ac_rows.extend(r)
            lrt_rows_all.extend(l)
            if r:
                # find best model AIC
                best = min(r, key=lambda x: x.get("AIC", np.inf))
                print(f"best={best['model']}  AIC={best['AIC']}")
            else:
                print("skipped")

    ac_df = pd.DataFrame(ac_rows)
    ac_df.to_csv(args.out_acoustic_models, index=False)
    print(f"\nSaved → {args.out_acoustic_models}")

    lrt_df = pd.DataFrame(lrt_rows_all)
    lrt_df.to_csv(args.out_acoustic_lrt, index=False)
    print(f"Saved → {args.out_acoustic_lrt}")

    # ── ICC summary ───────────────────────────────────────────────────────────
    null_rows = ac_df[ac_df["model"] == "M0_null"][
        ["phoneme", "formant", "ICC"]].copy()
    print(f"\nICC from null model (speaker random intercept):")
    print(null_rows.to_string(index=False))

    # ── neural LME ───────────────────────────────────────────────────────────
    print("\n── Neural LME ───────────────────────────────────────────────────")

    def load_reps(path):
        data = np.load(path, allow_pickle=True)
        reps = data["word_reps"]
        if reps.dtype == object:
            reps = np.stack(reps, axis=0)
        return reps.astype(float)[vowel_pos]

    neural_models = {
        "whisper": load_reps(args.whisper),
        "xlsr":    load_reps(args.xlsr),
    }

    neural_rows = []
    for model_name, reps in neural_models.items():
        for phoneme in vowels:
            print(f"  {model_name} / {phoneme} …", end=" ")
            r = run_neural_lme(df_v, reps, model_name, phoneme, args.n_pc)
            neural_rows.extend(r)
            if r:
                r2s = [x["R2_marginal"] for x in r]
                print(f"PC R²_m = {[round(x,3) for x in r2s]}")
            else:
                print("skipped")

    neural_df = pd.DataFrame(neural_rows)
    neural_df.to_csv(args.out_neural, index=False)
    print(f"\nSaved → {args.out_neural}")

    # ── R² comparison: acoustic vs neural for L1/L2 fixed effect ─────────────
    print("\n── Marginal R² comparison (L1 fixed effect) ─────────────────────")
    # acoustic: M1 main effects model
    ac_m1 = ac_df[ac_df["model"] == "M1_main"][
        ["phoneme", "formant", "R2_marginal", "R2_conditional"]].copy()
    ac_m1["representation"] = "acoustic"
    print("\nAcoustic (M1 main effects):")
    print(ac_m1.groupby("formant")[["R2_marginal", "R2_conditional"]]
          .mean().round(4).to_string())

    # neural: mean R² across PCs weighted by variance explained
    if not neural_df.empty:
        for mn in neural_models:
            sub = neural_df[neural_df["neural_model"] == mn]
            if sub.empty:
                continue
            # weighted mean R² across PCs
            weighted = (
                sub.groupby("phoneme")
                .apply(lambda g: np.average(
                    g["R2_marginal"].fillna(0),
                    weights=g["pct_var_explained"]
                ))
                .reset_index(name="R2_marginal_weighted")
            )
            print(f"\n{mn} (weighted mean R²_marginal across PCs):")
            print(weighted.to_string(index=False))

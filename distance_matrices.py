import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from normalise import is_vowel, lobanov_norm


# ── helpers ───────────────────────────────────────────────────────────────────

def upper_tri(M: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(M.shape[0], k=1)
    return M[idx]


def mantel_test(D1: np.ndarray, D2: np.ndarray,
                n_perm: int = 9999, seed: int = 42) -> dict:
    assert D1.shape == D2.shape, "Distance matrices must have the same shape"
    n = D1.shape[0]
    rng = np.random.default_rng(seed)
    v1, v2 = upper_tri(D1), upper_tri(D2)
    r_obs, _ = spearmanr(v1, v2)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = spearmanr(v1, upper_tri(D2[np.ix_(perm, perm)]))[0]
    p = ((np.abs(null) >= np.abs(r_obs)).sum() + 1) / (n_perm + 1)
    return {"r": float(r_obs), "p": float(p)}


# ── acoustic distance matrix (Euclidean only) ────────────────────────────────

def build_acoustic_matrix(df: pd.DataFrame,
                           phonemes: list) -> np.ndarray:
    """
    D_ac[i, j] = Euclidean distance between per-phoneme centroids
    in Lobanov-normalised (F1, F2) space.
    """
    centroids = np.array([
        df[df["phoneme"] == ph][["F1_normed", "F2_normed"]]
        .dropna().mean().to_numpy()
        for ph in phonemes
    ])                                              # (n_ph, 2)
    diff = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))        # (n_ph, n_ph)


# ── neural distance matrix (cosine) ──────────────────────────────────────────

def cosine_distance_matrix(mean_reps: np.ndarray) -> np.ndarray:
    """
    D[i, j] = 1 - cosine_similarity between per-phoneme mean embeddings.
    mean_reps: (n_ph, d)
    """
    norms = np.linalg.norm(mean_reps, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    Xn = mean_reps / norms
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim                                # (n_ph, n_ph)


def build_neural_matrix(df: pd.DataFrame,
                         reps: np.ndarray,
                         phonemes: list) -> np.ndarray:
    """
    Compute per-phoneme mean embeddings then return the cosine distance matrix.
    df and reps are aligned row-for-row.
    """
    mean_reps = np.array([
        reps[df["phoneme"] == ph].mean(axis=0)
        for ph in phonemes
    ])                                              # (n_ph, d)
    return cosine_distance_matrix(mean_reps)


# ── bootstrap CI on pairwise distances ───────────────────────────────────────

def bootstrap_ci_pair(df: pd.DataFrame,
                       reps_wh: np.ndarray,
                       reps_xl: np.ndarray,
                       ph_a: str, ph_b: str,
                       n_boot: int = 2000,
                       alpha: float = 0.05,
                       seed: int = 42) -> dict:
    """
    95% bootstrap CI on acoustic Euclidean, Whisper cosine, and XLS-R cosine
    distances between phonemes ph_a and ph_b.
    Resampling is at the speaker level (spec §10.2).
    """
    rng = np.random.default_rng(seed)
    speakers = df["spk"].unique()
    n_spk = len(speakers)

    eu_boot = np.empty(n_boot)
    wh_boot = np.empty(n_boot)
    xl_boot = np.empty(n_boot)

    for b in range(n_boot):
        sampled_spks = rng.choice(speakers, size=n_spk, replace=True)
        idx = np.concatenate([
            df.index[df["spk"] == s].to_numpy() for s in sampled_spks
        ])
        df_b   = df.loc[idx].reset_index(drop=True)
        wh_b   = reps_wh[idx]
        xl_b   = reps_xl[idx]

        mask_a = df_b["phoneme"] == ph_a
        mask_b = df_b["phoneme"] == ph_b

        # acoustic Euclidean
        ac_a = df_b.loc[mask_a, ["F1_normed", "F2_normed"]].dropna().to_numpy()
        ac_b = df_b.loc[mask_b, ["F1_normed", "F2_normed"]].dropna().to_numpy()
        if len(ac_a) < 1 or len(ac_b) < 1:
            eu_boot[b] = np.nan
        else:
            eu_boot[b] = np.linalg.norm(ac_a.mean(axis=0) - ac_b.mean(axis=0))

        def cosine_dist_pair(r, ma, mb):
            ra = r[ma.to_numpy()]
            rb = r[mb.to_numpy()]
            ra = ra[~np.isnan(ra).any(axis=1)]
            rb = rb[~np.isnan(rb).any(axis=1)]
            if len(ra) < 1 or len(rb) < 1:
                return np.nan
            ca = ra.mean(axis=0); cb = rb.mean(axis=0)
            na = np.linalg.norm(ca); nb = np.linalg.norm(cb)
            if na == 0 or nb == 0:
                return np.nan
            return float(1.0 - (ca @ cb) / (na * nb))

        wh_boot[b] = cosine_dist_pair(wh_b, mask_a, mask_b)
        xl_boot[b] = cosine_dist_pair(xl_b, mask_a, mask_b)

    lo, hi = alpha / 2, 1 - alpha / 2
    def ci(arr):
        obs = float(np.nanmean(arr))
        return obs, float(np.nanquantile(arr, lo)), float(np.nanquantile(arr, hi))

    eu_obs, eu_lo, eu_hi = ci(eu_boot)
    wh_obs, wh_lo, wh_hi = ci(wh_boot)
    xl_obs, xl_lo, xl_hi = ci(xl_boot)

    return {
        "phoneme_a": ph_a, "phoneme_b": ph_b,
        "ac_eu_obs":  round(eu_obs, 4),
        "ac_eu_ci_lo": round(eu_lo, 4), "ac_eu_ci_hi": round(eu_hi, 4),
        "wh_cos_obs":  round(wh_obs, 4),
        "wh_cos_ci_lo": round(wh_lo, 4), "wh_cos_ci_hi": round(wh_hi, 4),
        "xl_cos_obs":  round(xl_obs, 4),
        "xl_cos_ci_lo": round(xl_lo, 4), "xl_cos_ci_hi": round(xl_hi, 4),
    }


# ── matrix → long-form CSV ────────────────────────────────────────────────────

def matrix_to_df(D: np.ndarray, phonemes: list, dist_name: str) -> pd.DataFrame:
    rows = []
    for i, pi in enumerate(phonemes):
        for j, pj in enumerate(phonemes):
            rows.append({"phoneme_i": pi, "phoneme_j": pj,
                         dist_name: round(float(D[i, j]), 6)})
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic",    default="./reps/features_acoustic.csv")
    parser.add_argument("--whisper",     default="./reps/features_whisper.npz",
                        help="Full-dim Whisper reps (not PCA-reduced)")
    parser.add_argument("--xlsr",        default="./reps/features_xlsr.npz",
                        help="Full-dim XLS-R reps (not PCA-reduced)")
    parser.add_argument("--out-matrices",default="./tables/distance_matrices.csv")
    parser.add_argument("--out-mantel",  default="./tables/distance_mantel.csv")
    parser.add_argument("--out-ci",      default="./tables/distance_bootstrap_ci.csv")
    parser.add_argument("--pairs",       default="e-ɛ,o-ø,y-u",
                        help="Phoneme pairs for bootstrap CI, comma-separated, e.g. 'e-ɛ,o-ø,y-u'")
    parser.add_argument("--n-perm",      type=int, default=9999)
    parser.add_argument("--n-boot",      type=int, default=2000)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    # ── load acoustic ─────────────────────────────────────────────────────────
    df = pd.read_csv(args.acoustic)
    df["phoneme"] = df["phoneme"].fillna("")
    df = lobanov_norm(df)
    df_v = df[df["phoneme"].apply(is_vowel)].copy().reset_index(drop=True)

    vowels = sorted(df_v["phoneme"].unique())
    print(f"Vowels: {vowels}\n")

    # ── load neural reps, slice to vowel rows ─────────────────────────────────
    def load_reps(path: str) -> np.ndarray:
        data = np.load(path, allow_pickle=True)
        reps = data["word_reps"]
        if reps.dtype == object:
            reps = np.stack(reps, axis=0)
        return reps.astype(float)

    # reps are aligned to the full df rows; we need the vowel-row positions
    vowel_pos = df[df["phoneme"].apply(is_vowel)].index.to_numpy()

    reps_wh_full = load_reps(args.whisper)
    reps_xl_full = load_reps(args.xlsr)
    reps_wh = reps_wh_full[vowel_pos]   # (n_vowel_tokens, d_wh)
    reps_xl = reps_xl_full[vowel_pos]   # (n_vowel_tokens, d_xl)

    # ── build distance matrices ───────────────────────────────────────────────
    print("Building D_ac (Euclidean) …")
    D_ac = build_acoustic_matrix(df_v, vowels)

    print("Building D_wh (cosine) …")
    D_wh = build_neural_matrix(df_v, reps_wh, vowels)

    print("Building D_xl (cosine) …")
    D_xl = build_neural_matrix(df_v, reps_xl, vowels)

    # pretty-print matrices
    for name, D in [("D_ac (Euclidean)", D_ac),
                    ("D_wh (cosine)",    D_wh),
                    ("D_xl (cosine)",    D_xl)]:
        print(f"\n{name}:")
        print(pd.DataFrame(D, index=vowels, columns=vowels).round(4).to_string())

    # save combined long-form
    df_ac = matrix_to_df(D_ac, vowels, "euclidean")
    df_wh = matrix_to_df(D_wh, vowels, "whisper_cosine")
    df_xl = matrix_to_df(D_xl, vowels, "xlsr_cosine")
    combined = df_ac.merge(df_wh, on=["phoneme_i", "phoneme_j"]) \
                    .merge(df_xl, on=["phoneme_i", "phoneme_j"])
    combined.to_csv(args.out_matrices, index=False)
    print(f"\nSaved → {args.out_matrices}")

    # ── Mantel tests ──────────────────────────────────────────────────────────
    pairs_mantel = [
        ("D_ac", "D_wh", D_ac, D_wh),
        ("D_ac", "D_xl", D_ac, D_xl),
        ("D_wh", "D_xl", D_wh, D_xl),
    ]
    print(f"\nMantel tests (B={args.n_perm}) …")
    mantel_rows = []
    for n1, n2, Da, Db in pairs_mantel:
        res = mantel_test(Da, Db, n_perm=args.n_perm, seed=args.seed)
        print(f"  {n1} vs {n2}: r={res['r']:.4f}  p={res['p']:.4f}")
        mantel_rows.append({"D1": n1, "D2": n2,
                             "mantel_r": round(res["r"], 4),
                             "p_value":  round(res["p"], 4),
                             "n_perm": args.n_perm,
                             "n_phonemes": len(vowels)})
    mantel_df = pd.DataFrame(mantel_rows)
    mantel_df.to_csv(args.out_mantel, index=False)
    print(f"Saved → {args.out_mantel}")

    # ── bootstrap CIs ─────────────────────────────────────────────────────────
    pairs_boot = [tuple(p.split("-")) for p in args.pairs.split(",")]
    pairs_boot = [(a, b) for a, b in pairs_boot
                  if a in vowels and b in vowels]

    if not pairs_boot:
        print(f"\nNo valid pairs found. Available vowels: {vowels}")
        print("Adjust --pairs (e.g. --pairs 'e-ɛ,o-ø,y-u')")
    else:
        print(f"\nBootstrap CIs (B={args.n_boot}, speaker-level) …")
        ci_rows = []
        for ph_a, ph_b in pairs_boot:
            row = bootstrap_ci_pair(df_v, reps_wh, reps_xl, ph_a, ph_b,
                                    n_boot=args.n_boot, seed=args.seed)
            ci_rows.append(row)
            print(f"  {ph_a}–{ph_b}:")
            print(f"    AC  EU : {row['ac_eu_obs']:.4f} "
                  f"[{row['ac_eu_ci_lo']:.4f}, {row['ac_eu_ci_hi']:.4f}]")
            print(f"    WH cos : {row['wh_cos_obs']:.4f} "
                  f"[{row['wh_cos_ci_lo']:.4f}, {row['wh_cos_ci_hi']:.4f}]")
            print(f"    XL cos : {row['xl_cos_obs']:.4f} "
                  f"[{row['xl_cos_ci_lo']:.4f}, {row['xl_cos_ci_hi']:.4f}]")
        ci_df = pd.DataFrame(ci_rows)
        ci_df.to_csv(args.out_ci, index=False)
        print(f"Saved → {args.out_ci}")

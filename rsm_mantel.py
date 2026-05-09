import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from normalise import is_vowel, lobanov_norm


# ── RSM builders ────────────────────────────────────────────────────────────

def acoustic_rsm(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Negative Euclidean distance between (F1_normed, F2_normed) token vectors.
    Returns (S, valid_idx) where S is an (n, n) RSM over the valid rows.
    """
    vowel_mask = df["phoneme"].apply(is_vowel)
    sub = df[vowel_mask].copy()
    X = sub[["F1_normed", "F2_normed"]].to_numpy(dtype=float)

    # keep only fully observed rows
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    idx = sub.index[valid]

    D = np.linalg.norm(X[:, None] - X[None, :], axis=-1)   # Euclidean distance
    S = -D                                                   # negative → similarity
    return S, idx


def cosine_rsm(X: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix for an (n, d) array."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    Xn = X / norms
    return Xn @ Xn.T


def neural_rsm(npz_path: str, valid_idx: np.ndarray) -> np.ndarray:
    """
    Load full-dim neural reps, slice to valid_idx rows, return cosine RSM.
    valid_idx is a positional integer array into the original metadata rows.
    """
    data = np.load(npz_path, allow_pickle=True)
    reps = data["word_reps"]
    if reps.dtype == object:
        reps = np.stack(reps, axis=0)

    reps = reps[valid_idx]

    # drop any NaN rows (should be rare after acoustic filtering)
    nan_mask = np.isnan(reps).any(axis=1)
    if nan_mask.any():
        reps = reps[~nan_mask]

    return cosine_rsm(reps.astype(float))


# ── Mantel test ──────────────────────────────────────────────────────────────

def upper_tri(M: np.ndarray) -> np.ndarray:
    """Return the upper-triangular elements (excluding diagonal) as a 1D array."""
    idx = np.triu_indices(M.shape[0], k=1)
    return M[idx]


def mantel_test(S1: np.ndarray, S2: np.ndarray,
                n_perm: int = 9999,
                seed: int = 42) -> dict:
    """
    Mantel test: Spearman rank correlation between upper triangles of two RSMs.
    Permutation p-value: rows/cols of S2 are permuted simultaneously.
    Both matrices must be the same shape.
    """
    assert S1.shape == S2.shape, "RSMs must have the same shape"
    n = S1.shape[0]
    rng = np.random.default_rng(seed)

    v1 = upper_tri(S1)
    v2 = upper_tri(S2)

    r_obs, _ = spearmanr(v1, v2)

    # permutation null
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        S2p = S2[np.ix_(perm, perm)]
        r_null, _ = spearmanr(v1, upper_tri(S2p))
        null[i] = r_null

    p_value = ((np.abs(null) >= np.abs(r_obs)).sum() + 1) / (n_perm + 1)
    return {"r": float(r_obs), "p": float(p_value), "n_perm": n_perm}


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic",  default="./reps/features_acoustic.csv",
                        help="CSV output of extract_acoustics.py")
    parser.add_argument("--whisper",   default="./reps/features_whisper.npz",
                        help="Full-dim Whisper reps (not PCA-reduced)")
    parser.add_argument("--xlsr",      default="./reps/features_xlsr.npz",
                        help="Full-dim XLS-R reps (not PCA-reduced)")
    parser.add_argument("--n-perm",    type=int, default=9999)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    default="./tables/rsm_mantel.csv")
    args = parser.parse_args()

    # ── load & normalise acoustic features ──────────────────────────────────
    df = pd.read_csv(args.acoustic)
    df["phoneme"] = df["phoneme"].fillna("")
    df = lobanov_norm(df)

    # ── build acoustic RSM (vowels only, valid rows) ─────────────────────────
    S_ac, valid_idx = acoustic_rsm(df)
    # valid_idx are pandas index labels; convert to positional for numpy slicing
    valid_pos = df.index.get_indexer(valid_idx)

    print(f"Acoustic RSM: {S_ac.shape[0]} tokens")

    # ── build neural RSMs ────────────────────────────────────────────────────
    # We need to handle the case where neural reps have NaN rows differently:
    # after slicing to valid_pos, both neural RSMs must share the same token set
    # as the acoustic RSM. We do a joint NaN filter.

    def load_neural(path: str) -> np.ndarray:
        data = np.load(path, allow_pickle=True)
        reps = data["word_reps"]
        if reps.dtype == object:
            reps = np.stack(reps, axis=0)
        return reps.astype(float)

    wh_all = load_neural(args.whisper)[valid_pos]
    xl_all = load_neural(args.xlsr)[valid_pos]

    # joint valid mask (no NaN in either model)
    joint_valid = (
        ~np.isnan(wh_all).any(axis=1) &
        ~np.isnan(xl_all).any(axis=1)
    )

    S_ac = S_ac[np.ix_(joint_valid, joint_valid)]
    wh   = wh_all[joint_valid]
    xl   = xl_all[joint_valid]

    S_wh = cosine_rsm(wh)
    S_xl = cosine_rsm(xl)

    n_tokens = S_ac.shape[0]
    print(f"Joint valid tokens: {n_tokens}")
    print(f"Running Mantel tests with {args.n_perm} permutations …")

    # ── Mantel tests ─────────────────────────────────────────────────────────
    pairs = [
        ("acoustic", "whisper", S_ac, S_wh),
        ("acoustic", "xlsr",    S_ac, S_xl),
        ("whisper",  "xlsr",    S_wh, S_xl),
    ]

    rows = []
    for name1, name2, Sa, Sb in pairs:
        res = mantel_test(Sa, Sb, n_perm=args.n_perm, seed=args.seed)
        row = {
            "RSM_1":   name1,
            "RSM_2":   name2,
            "mantel_r": round(res["r"], 4),
            "p_value":  round(res["p"], 4),
            "n_perm":   res["n_perm"],
            "n_tokens": n_tokens,
            "significant_0.05": res["p"] < 0.05,
        }
        rows.append(row)
        print(f"  {name1} vs {name2}: r={res['r']:.4f}, p={res['p']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")

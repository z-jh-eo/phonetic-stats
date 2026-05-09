import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from normalise import is_vowel, lobanov_norm


# ── phoneme class definitions ─────────────────────────────────────────────────

VOWELS = {"a", "e", "ɛ", "i", "o", "ø", "u", "y"}

# ≥6 consonants covering different manner and place classes (spec §9.2)
CONSONANTS = {
    "p",   # bilabial stop voiceless
    "b",   # bilabial stop voiced
    "f",   # labiodental fricative voiceless
    "v",   # labiodental fricative voiced
    "s",   # alveolar fricative voiceless
    "z",   # alveolar fricative voiced
    "ʃ",   # postalveolar fricative voiceless
    "ʒ",   # postalveolar fricative voiced
    "t",   # alveolar stop voiceless
    "d",   # alveolar stop voiced
    "k",   # velar stop voiceless
    "g",   # velar stop voiced
    "m",   # bilabial nasal
    "n",   # alveolar nasal
    "l",   # lateral approximant
    "ʁ",   # uvular fricative / approximant  (French R)
}

TARGET_PHONEMES = VOWELS | CONSONANTS

# ground-truth partitions
MANNER = {
    # vowels
    "a": "vowel", "e": "vowel", "ɛ": "vowel", "i": "vowel",
    "o": "vowel", "ø": "vowel", "u": "vowel", "y": "vowel",
    # stops
    "p": "stop", "b": "stop", "t": "stop", "d": "stop",
    "k": "stop", "g": "stop",
    # fricatives
    "f": "fricative", "v": "fricative", "s": "fricative",
    "z": "fricative", "ʃ": "fricative", "ʒ": "fricative", "ʁ": "fricative",
    # nasals
    "m": "nasal", "n": "nasal",
    # approximants / sonorants
    "l": "lateral",
}

VOICING = {
    "a": "voiced", "e": "voiced", "ɛ": "voiced", "i": "voiced",
    "o": "voiced", "ø": "voiced", "u": "voiced", "y": "voiced",
    "b": "voiced", "d": "voiced", "g": "voiced",
    "v": "voiced", "z": "voiced", "ʒ": "voiced",
    "m": "voiced", "n": "voiced", "l": "voiced", "ʁ": "voiced",
    "p": "voiceless", "t": "voiceless", "k": "voiceless",
    "f": "voiceless", "s": "voiceless", "ʃ": "voiceless",
}

CV_CLASS = {ph: ("vowel" if ph in VOWELS else "consonant")
            for ph in TARGET_PHONEMES}


def ground_truth_labels(phonemes: list, partition: dict) -> np.ndarray:
    cats = sorted(set(partition.values()))
    cat2int = {c: i for i, c in enumerate(cats)}
    return np.array([cat2int.get(p, -1) for p in phonemes])


# ── acoustic feature matrix ───────────────────────────────────────────────────

def build_acoustic_features(df: pd.DataFrame,
                              phonemes: list) -> tuple[np.ndarray, list[str]]:
    """
    Per-phoneme mean feature vector:
      - Vowels   : F1_normed, F2_normed, dur_ms
      - Consonants: dur_ms, SCG (spectral centre of gravity)

    All features are z-scored across phonemes before clustering so that
    duration and SCG (Hz range) are on the same scale as Lobanov F1/F2.
    Returns (X, feature_names).
    """
    feat_names = ["F1_normed", "F2_normed", "dur_ms", "scg"]
    rows = []
    for ph in phonemes:
        sub = df[df["phoneme"] == ph]
        f1  = float(sub["F1_normed"].dropna().mean()) \
              if ph in VOWELS else np.nan
        f2  = float(sub["F2_normed"].dropna().mean()) \
              if ph in VOWELS else np.nan
        dur = float(sub["dur"].dropna().mean() * 1000) \
              if "dur" in sub.columns else np.nan   # seconds → ms
        # SCG stored in signal_rep tuple index 13; or as a separate column
        # Try "scg" column first, else NaN
        scg = float(sub["scg"].dropna().mean()) \
              if "scg" in sub.columns else np.nan
        rows.append([f1, f2, dur, scg])

    X = np.array(rows, dtype=float)   # (n_ph, 4)

    # z-score column-wise (ignore NaN columns)
    scaler = StandardScaler()
    # fill NaN with column mean before scaling, then restore NaN
    X_filled = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_m = np.isnan(col)
        if nan_m.all():
            continue
        col_mean = np.nanmean(col)
        X_filled[nan_m, j] = col_mean
    X_scaled = scaler.fit_transform(X_filled)
    # restore NaN where all values were NaN for that column
    for j in range(X.shape[1]):
        if np.isnan(X[:, j]).all():
            X_scaled[:, j] = 0.0   # treat as zero-contribution
    return X_scaled, feat_names


# ── neural mean vectors ───────────────────────────────────────────────────────

def neural_mean_vectors(df: pd.DataFrame, reps: np.ndarray,
                         phonemes: list) -> np.ndarray:
    return np.array([
        reps[df["phoneme"].to_numpy() == ph].mean(axis=0)
        for ph in phonemes
    ])


# ── distance matrices ─────────────────────────────────────────────────────────

def euclidean_dist_matrix(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def cosine_dist_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    Xn  = X / norms
    sim = np.clip(Xn @ Xn.T, -1.0, 1.0)
    return 1.0 - sim


# ── Ward clustering + evaluation ──────────────────────────────────────────────

def cluster_and_evaluate(D: np.ndarray, phonemes: list,
                          k_values: list,
                          partitions: dict) -> dict:
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="ward")

    results = {"linkage": Z, "phonemes": phonemes,
               "silhouette": {}, "ari": {}}

    for k in k_values:
        labels_pred = fcluster(Z, k, criterion="maxclust") - 1
        if 1 < k < len(phonemes) and len(set(labels_pred)) > 1:
            try:
                sil = float(silhouette_score(D, labels_pred,
                                             metric="precomputed"))
            except Exception:
                sil = np.nan
        else:
            sil = np.nan
        results["silhouette"][k] = round(sil, 4) if not np.isnan(sil) else np.nan

        results["ari"][k] = {}
        for part_name, part_dict in partitions.items():
            gt    = ground_truth_labels(phonemes, part_dict)
            valid = gt >= 0
            if valid.sum() < 2:
                results["ari"][k][part_name] = np.nan
                continue
            ari = adjusted_rand_score(gt[valid], labels_pred[valid])
            results["ari"][k][part_name] = round(float(ari), 4)

    return results


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_dendrogram(Z, phonemes, title, outpath,
                    cv_class=None):
    """Colour leaf labels by C/V class if cv_class dict provided."""
    fig, ax = plt.subplots(figsize=(max(8, len(phonemes) * 0.65), 4.5))
    ddata = dendrogram(Z, labels=phonemes, ax=ax,
                       color_threshold=0.7 * max(Z[:, 2]),
                       no_plot=False)
    if cv_class:
        for lbl in ax.get_xticklabels():
            ph  = lbl.get_text()
            col = "#c0392b" if cv_class.get(ph) == "vowel" else "#2980b9"
            lbl.set_color(col)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Ward distance", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(color="#c0392b", label="vowel"),
               Patch(color="#2980b9", label="consonant")]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath}")


def plot_silhouette(results_all, k_values, outpath):
    fig, ax = plt.subplots(figsize=(7, 4))
    for rep_name, res in results_all.items():
        sil = [res["silhouette"].get(k, np.nan) for k in k_values]
        ax.plot(k_values, sil, marker="o", label=rep_name)
    ax.set_xlabel("k", fontsize=10)
    ax.set_ylabel("Silhouette coefficient", fontsize=10)
    ax.set_title("Silhouette — consonant+vowel clustering", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath}")


def plot_ari_heatmap(ari_df, outpath):
    fig, ax = plt.subplots(figsize=(max(6, len(ari_df.columns) * 1.3),
                                    max(3, len(ari_df) * 0.55)))
    im = ax.imshow(ari_df.values.astype(float),
                   vmin=-0.1, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(ari_df.columns)))
    ax.set_xticklabels(ari_df.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(ari_df.index)))
    ax.set_yticklabels(ari_df.index, fontsize=9)
    for i in range(len(ari_df.index)):
        for j in range(len(ari_df.columns)):
            v = float(ari_df.values[i, j])
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8)
    plt.colorbar(im, ax=ax, label="ARI")
    ax.set_title("ARI — consonant+vowel clustering", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic", default="./reps/features_acoustic.csv")
    parser.add_argument("--whisper",  default="./reps/features_whisper.npz")
    parser.add_argument("--xlsr",     default="./reps/features_xlsr.npz")
    parser.add_argument("--k-values", default="2,3,4,5,6,7,8")
    parser.add_argument("--outdir",   default="./plots")
    parser.add_argument("--out-ari",  default="./tables/cv_clustering_ari.csv")
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_ari), exist_ok=True)

    k_values = [int(k) for k in args.k_values.split(",")]

    partitions = {
        "C_vs_V":  CV_CLASS,
        "manner":  MANNER,
        "voicing": VOICING,
    }

    # ── load ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(args.acoustic)
    df["phoneme"] = df["phoneme"].fillna("")
    df = lobanov_norm(df)

    # keep only target phonemes present in data
    present = set(df["phoneme"].unique())
    phonemes = sorted([p for p in TARGET_PHONEMES if p in present])
    print(f"Target phonemes present in data: {phonemes}")
    print(f"  Vowels    : {[p for p in phonemes if p in VOWELS]}")
    print(f"  Consonants: {[p for p in phonemes if p in CONSONANTS]}\n")

    if len(phonemes) < 4:
        raise ValueError("Too few target phonemes found. "
                         "Check phoneme labels in your corpus.")

    vowel_and_cons_pos = df[df["phoneme"].isin(phonemes)].index.to_numpy()
    df_tc = df[df["phoneme"].isin(phonemes)].copy().reset_index(drop=True)

    def load_reps(path):
        data = np.load(path, allow_pickle=True)
        reps = data["word_reps"]
        if reps.dtype == object:
            reps = np.stack(reps, axis=0)
        return reps.astype(float)[vowel_and_cons_pos]

    reps_wh = load_reps(args.whisper)
    reps_xl = load_reps(args.xlsr)

    # ── build feature matrices ────────────────────────────────────────────────
    X_ac, feat_names = build_acoustic_features(df_tc, phonemes)
    print(f"Acoustic features: {feat_names}")

    X_wh = neural_mean_vectors(df_tc, reps_wh, phonemes)
    X_xl = neural_mean_vectors(df_tc, reps_xl, phonemes)

    # ── distance matrices ─────────────────────────────────────────────────────
    D_ac = euclidean_dist_matrix(X_ac)
    D_wh = cosine_dist_matrix(X_wh)
    D_xl = cosine_dist_matrix(X_xl)

    representations = {
        "acoustic (F1/F2+dur+SCG)": D_ac,
        "whisper (cosine)":          D_wh,
        "xlsr (cosine)":             D_xl,
    }

    # ── cluster + evaluate ────────────────────────────────────────────────────
    all_results = {}
    ari_rows    = []

    for rep_name, D in representations.items():
        print(f"── {rep_name} ─────────────────────────────────────────────────")
        res = cluster_and_evaluate(D, phonemes, k_values, partitions)
        all_results[rep_name] = res

        slug = rep_name.split()[0]
        plot_dendrogram(
            res["linkage"], phonemes,
            title=f"C+V dendrogram — {rep_name}",
            outpath=f"{args.outdir}/dendrogram_cv_{slug}.png",
            cv_class=CV_CLASS,
        )

        print(f"  {'k':>3}  {'silhouette':>10}  " +
              "  ".join(f"{p:>10}" for p in partitions))
        for k in k_values:
            sil = res["silhouette"].get(k, np.nan)
            ari_str = "  ".join(
                f"{res['ari'][k].get(p, np.nan):>10.4f}" for p in partitions)
            print(f"  {k:>3}  {sil:>10.4f}  {ari_str}")
            row = {"representation": rep_name, "k": k, "silhouette": sil}
            row.update({f"ARI_{p}": res["ari"][k].get(p, np.nan)
                        for p in partitions})
            ari_rows.append(row)
        print()

    # ── silhouette plot ───────────────────────────────────────────────────────
    plot_silhouette(all_results, k_values,
                    outpath=f"{args.outdir}/silhouette_cv.png")

    # ── ARI heatmap at best k per representation ──────────────────────────────
    ari_df = pd.DataFrame(ari_rows)
    ari_df.to_csv(args.out_ari, index=False)
    print(f"Saved → {args.out_ari}")

    best_rows = []
    for rep_name in representations:
        sub    = ari_df[ari_df["representation"] == rep_name]
        best_k = sub.loc[sub["silhouette"].idxmax(), "k"]
        best   = sub[sub["k"] == best_k].iloc[0]
        row    = {"representation": rep_name, "best_k": int(best_k),
                  "silhouette": round(float(best["silhouette"]), 4)}
        for p in partitions:
            row[f"ARI_{p}"] = round(float(best[f"ARI_{p}"]), 4)
        best_rows.append(row)
        print(f"{rep_name}: best k={best_k}  "
              + "  ".join(f"ARI_{p}={row[f'ARI_{p}']:.4f}"
                          for p in partitions))

    heatmap_df = pd.DataFrame(best_rows).set_index("representation")[
        [f"ARI_{p}" for p in partitions]
    ]
    heatmap_df.columns = list(partitions.keys())
    plot_ari_heatmap(heatmap_df,
                     outpath=f"{args.outdir}/ari_heatmap_cv.png")

    # ── C/V boundary analysis ─────────────────────────────────────────────────
    print("\n── C/V boundary recovery (k=2) ─────────────────────────────────")
    for rep_name, res in all_results.items():
        ari_cv = res["ari"].get(2, {}).get("C_vs_V", np.nan)
        labels = fcluster(res["linkage"], 2, criterion="maxclust") - 1
        cluster0 = [phonemes[i] for i, l in enumerate(labels) if l == 0]
        cluster1 = [phonemes[i] for i, l in enumerate(labels) if l == 1]
        print(f"\n  {rep_name}  ARI(C/V, k=2)={ari_cv:.4f}")
        print(f"    Cluster 0: {cluster0}")
        print(f"    Cluster 1: {cluster1}")

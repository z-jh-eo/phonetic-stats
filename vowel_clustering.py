import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from normalise import is_vowel, lobanov_norm


# ── ground-truth partitions (French vowel trapezoid) ─────────────────────────

# Front / back distinction
FRONT_BACK = {
    "i": "front", "e": "front", "ɛ": "front", "a": "front",
    "y": "front", "ø": "front",
    "u": "back",  "o": "back",
}

# High / mid / low distinction
HEIGHT = {
    "i": "high", "y": "high", "u": "high",
    "e": "mid",  "ø": "mid",  "o": "mid",  "ɛ": "mid",
    "a": "low",
}

# Rounded / unrounded
ROUNDING = {
    "i": "unrounded", "e": "unrounded", "ɛ": "unrounded", "a": "unrounded",
    "y": "rounded",   "ø": "rounded",   "u": "rounded",   "o": "rounded",
}


def ground_truth_labels(phonemes: list, partition: dict) -> np.ndarray:
    """Map phoneme list to integer cluster labels via partition dict."""
    cats = sorted(set(partition.values()))
    cat2int = {c: i for i, c in enumerate(cats)}
    return np.array([cat2int.get(partition.get(p), -1) for p in phonemes])



# ── mean-vector representations per phoneme ───────────────────────────────────

def acoustic_mean_vectors(df: pd.DataFrame, phonemes: list) -> np.ndarray:
    """(n_ph, 2) array of per-phoneme Lobanov centroids."""
    return np.array([
        df[df["phoneme"] == ph][["F1_normed", "F2_normed"]].dropna().mean().to_numpy()
        for ph in phonemes
    ])


def neural_mean_vectors(df: pd.DataFrame, reps: np.ndarray,
                         phonemes: list) -> np.ndarray:
    """(n_ph, d) array of per-phoneme mean embeddings."""
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
                          partitions: dict[str, dict]) -> dict:
    """
    Ward hierarchical clustering on a precomputed distance matrix D.
    Returns linkage matrix, silhouette scores, ARI scores per k and partition.
    """
    # scipy Ward requires condensed distance matrix
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="ward")

    results = {"linkage": Z, "phonemes": phonemes,
               "silhouette": {}, "ari": {}}

    for k in k_values:
        labels_pred = fcluster(Z, k, criterion="maxclust") - 1  # 0-indexed

        # silhouette (needs at least 2 clusters and more points than clusters)
        if 1 < k < len(phonemes) and len(set(labels_pred)) > 1:
            try:
                sil = float(silhouette_score(D, labels_pred, metric="precomputed"))
            except Exception:
                sil = np.nan
        else:
            sil = np.nan
        results["silhouette"][k] = round(sil, 4) if not np.isnan(sil) else np.nan

        # ARI against each ground-truth partition
        results["ari"][k] = {}
        for part_name, part_dict in partitions.items():
            gt = ground_truth_labels(phonemes, part_dict)
            valid = gt >= 0
            if valid.sum() < 2:
                results["ari"][k][part_name] = np.nan
                continue
            ari = adjusted_rand_score(gt[valid], labels_pred[valid])
            results["ari"][k][part_name] = round(float(ari), 4)

    return results


# ── dendrogram plot ───────────────────────────────────────────────────────────

def plot_dendrogram(Z: np.ndarray, phonemes: list,
                    title: str, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(phonemes) * 0.8), 4))
    dendrogram(Z, labels=phonemes, ax=ax,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Ward linkage distance", fontsize=9)
    ax.set_xlabel("Phoneme", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath}")


# ── silhouette plot ───────────────────────────────────────────────────────────

def plot_silhouette(results_all: dict[str, dict],
                    k_values: list, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for rep_name, res in results_all.items():
        sil_vals = [res["silhouette"].get(k, np.nan) for k in k_values]
        ax.plot(k_values, sil_vals, marker="o", label=rep_name)
    ax.set_xlabel("Number of clusters k", fontsize=10)
    ax.set_ylabel("Silhouette coefficient", fontsize=10)
    ax.set_title("Silhouette scores — vowel clustering", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath}")


# ── ARI heatmap ───────────────────────────────────────────────────────────────

def plot_ari_table(ari_df: pd.DataFrame, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(ari_df.columns) * 1.2),
                                    max(3, len(ari_df) * 0.5)))
    im = ax.imshow(ari_df.values.astype(float), vmin=-0.1, vmax=1.0,
                   cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(ari_df.columns)))
    ax.set_xticklabels(ari_df.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(ari_df.index)))
    ax.set_yticklabels(ari_df.index, fontsize=9)
    for i in range(len(ari_df.index)):
        for j in range(len(ari_df.columns)):
            val = ari_df.values[i, j]
            if not np.isnan(float(val)):
                ax.text(j, i, f"{float(val):.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="ARI")
    ax.set_title("Adjusted Rand Index — vowel clustering vs ground truth", fontsize=10)
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
    parser.add_argument("--k-values", default="2,3,4,5,6",
                        help="Comma-separated k values to evaluate")
    parser.add_argument("--outdir",   default="./plots")
    parser.add_argument("--out-ari",  default="./tables/vowel_clustering_ari.csv")
    parser.add_argument("--out-sil",  default="./tables/vowel_clustering_silhouette.csv")
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_ari), exist_ok=True)

    k_values = [int(k) for k in args.k_values.split(",")]

    partitions = {
        "front_back": FRONT_BACK,
        "height":     HEIGHT,
        "rounding":   ROUNDING,
    }

    # ── load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(args.acoustic)
    df["phoneme"] = df["phoneme"].fillna("")
    df = lobanov_norm(df)
    df_v = df[df["phoneme"].apply(is_vowel)].copy().reset_index(drop=True)

    vowel_pos = df[df["phoneme"].apply(is_vowel)].index.to_numpy()
    vowels    = sorted(df_v["phoneme"].unique())
    print(f"Vowels: {vowels}\n")

    def load_reps(path):
        data = np.load(path, allow_pickle=True)
        reps = data["word_reps"]
        if reps.dtype == object:
            reps = np.stack(reps, axis=0)
        return reps.astype(float)[vowel_pos]

    reps_wh = load_reps(args.whisper)
    reps_xl = load_reps(args.xlsr)

    # ── mean vectors per phoneme ──────────────────────────────────────────────
    X_ac = acoustic_mean_vectors(df_v, vowels)
    X_wh = neural_mean_vectors(df_v, reps_wh, vowels)
    X_xl = neural_mean_vectors(df_v, reps_xl, vowels)

    # ── distance matrices ─────────────────────────────────────────────────────
    D_ac = euclidean_dist_matrix(X_ac)
    D_wh = cosine_dist_matrix(X_wh)
    D_xl = cosine_dist_matrix(X_xl)

    representations = {
        "acoustic (Euclidean)": D_ac,
        "whisper (cosine)":     D_wh,
        "xlsr (cosine)":        D_xl,
    }

    # ── cluster + evaluate ────────────────────────────────────────────────────
    all_results = {}
    ari_rows    = []
    sil_rows    = []

    for rep_name, D in representations.items():
        print(f"── {rep_name} ────────────────────────────────────────────────")
        res = cluster_and_evaluate(D, vowels, k_values, partitions)
        all_results[rep_name] = res

        # dendrogram
        slug = rep_name.split()[0]
        plot_dendrogram(
            res["linkage"], vowels,
            title=f"Vowel dendrogram — {rep_name}",
            outpath=f"{args.outdir}/dendrogram_vowels_{slug}.png"
        )

        # print ARI table
        print(f"  {'k':>3}  {'silhouette':>10}  " +
              "  ".join(f"{p:>12}" for p in partitions))
        for k in k_values:
            sil = res["silhouette"].get(k, np.nan)
            ari_vals = "  ".join(
                f"{res['ari'][k].get(p, np.nan):>12.4f}"
                for p in partitions
            )
            print(f"  {k:>3}  {sil:>10.4f}  {ari_vals}")

            # collect for CSV
            row = {"representation": rep_name, "k": k,
                   "silhouette": sil}
            row.update({f"ARI_{p}": res["ari"][k].get(p, np.nan)
                        for p in partitions})
            ari_rows.append(row)
            sil_rows.append({"representation": rep_name, "k": k,
                              "silhouette": sil})
        print()

    # ── silhouette plot ───────────────────────────────────────────────────────
    plot_silhouette(all_results, k_values,
                    outpath=f"{args.outdir}/silhouette_vowels.png")

    # ── ARI heatmap: best k per representation ────────────────────────────────
    # find k with best mean ARI across partitions for each representation
    ari_df_full = pd.DataFrame(ari_rows)
    ari_df_full.to_csv(args.out_ari, index=False)
    print(f"Saved → {args.out_ari}")

    sil_df = pd.DataFrame(sil_rows)
    sil_df.to_csv(args.out_sil, index=False)
    print(f"Saved → {args.out_sil}")

    # best-k ARI table for the heatmap (pick k with highest silhouette)
    best_k_rows = []
    for rep_name in representations:
        sub = ari_df_full[ari_df_full["representation"] == rep_name].copy()
        best_k = sub.loc[sub["silhouette"].idxmax(), "k"]
        best   = sub[sub["k"] == best_k].iloc[0]
        row = {"representation": rep_name, "best_k": int(best_k),
               "silhouette": round(float(best["silhouette"]), 4)}
        for p in partitions:
            row[f"ARI_{p}"] = round(float(best[f"ARI_{p}"]), 4)
        best_k_rows.append(row)
        print(f"{rep_name}: best k={best_k}  "
              + "  ".join(f"ARI_{p}={row[f'ARI_{p}']:.4f}"
                          for p in partitions))

    best_k_df = pd.DataFrame(best_k_rows)
    ari_heatmap_df = best_k_df.set_index("representation")[
        [f"ARI_{p}" for p in partitions]
    ]
    ari_heatmap_df.columns = list(partitions.keys())
    plot_ari_table(ari_heatmap_df,
                   outpath=f"{args.outdir}/ari_heatmap_vowels.png")

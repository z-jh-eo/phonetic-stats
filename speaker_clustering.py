import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from normalise import is_vowel, lobanov_norm


# ── speaker representation vectors ───────────────────────────────────────────

def speaker_acoustic_vectors(df: pd.DataFrame,
                              vowels: list) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Each speaker → concatenation of per-vowel mean [F1_normed, F2_normed].
    Returns (speaker_meta_df, X) where X is (n_speakers, n_vowels*2).
    """
    speakers = sorted(df["spk"].unique())
    rows = []
    meta = []

    for spk in speakers:
        spk_df = df[df["spk"] == spk]
        vec = []
        for ph in vowels:
            sub = spk_df[spk_df["phoneme"] == ph][["F1_normed", "F2_normed"]].dropna()
            if len(sub) == 0:
                vec.extend([np.nan, np.nan])
            else:
                vec.extend(sub.mean().tolist())
        rows.append(vec)
        # speaker metadata: take first row
        first = spk_df.iloc[0]
        meta.append({
            "spk":    spk,
            "L1":     first["L1"],
            "Gender": first["Gender"],
        })

    X = np.array(rows, dtype=float)
    meta_df = pd.DataFrame(meta)
    return meta_df, X


def speaker_neural_vectors(df: pd.DataFrame, reps: np.ndarray,
                            vowels: list) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Each speaker → concatenation of per-vowel mean neural embeddings.
    Returns (speaker_meta_df, X) where X is (n_speakers, n_vowels*d).
    """
    speakers = sorted(df["spk"].unique())
    rows = []
    meta = []

    for spk in speakers:
        spk_mask = df["spk"] == spk
        spk_df   = df[spk_mask].reset_index(drop=True)
        spk_reps = reps[spk_mask.to_numpy()]
        vec = []
        for ph in vowels:
            ph_mask = spk_df["phoneme"] == ph
            r = spk_reps[ph_mask.to_numpy()]
            r = r[~np.isnan(r).any(axis=1)]
            if len(r) == 0:
                # fill with zeros — will be handled by imputation below
                d = reps.shape[1]
                vec.extend([0.0] * d)
            else:
                vec.extend(r.mean(axis=0).tolist())
        rows.append(vec)
        first = spk_df.iloc[0]
        meta.append({"spk": spk, "L1": first["L1"], "Gender": first["Gender"]})

    X = np.array(rows, dtype=float)
    meta_df = pd.DataFrame(meta)
    return meta_df, X


# ── impute NaN columns ────────────────────────────────────────────────────────

def impute_colmean(X: np.ndarray) -> np.ndarray:
    """Replace NaN with column mean; drop all-NaN columns."""
    X = X.copy()
    col_means = np.nanmean(X, axis=0)
    nan_mask  = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    # drop columns that are still NaN (all-NaN col)
    valid_cols = ~np.isnan(X).any(axis=0)
    return X[:, valid_cols]


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

def cluster_and_evaluate(D: np.ndarray, meta_df: pd.DataFrame,
                          k_values: list) -> dict:
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="ward")

    results = {"linkage": Z, "silhouette": {}, "ari": {}}
    partitions = {"L1": meta_df["L1"].to_numpy(),
                  "Gender": meta_df["Gender"].to_numpy()}

    for k in k_values:
        labels_pred = fcluster(Z, k, criterion="maxclust") - 1
        n_spk = len(meta_df)

        if 1 < k < n_spk and len(set(labels_pred)) > 1:
            try:
                sil = float(silhouette_score(D, labels_pred,
                                             metric="precomputed"))
            except Exception:
                sil = np.nan
        else:
            sil = np.nan
        results["silhouette"][k] = round(sil, 4) if not np.isnan(sil) else np.nan

        results["ari"][k] = {}
        for part_name, gt in partitions.items():
            ari = adjusted_rand_score(gt, labels_pred)
            results["ari"][k][part_name] = round(float(ari), 4)

    return results


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_dendrogram(Z, labels, meta_df, title, outpath):
    """
    Dendrogram with leaf labels = spk_id, coloured by L1 status.
    """
    l1_vals  = meta_df["L1"].tolist()
    l1_uniq  = sorted(set(l1_vals))
    palette  = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad"]
    l1_color = {v: palette[i] for i, v in enumerate(l1_uniq)}

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    ddata = dendrogram(Z, labels=labels, ax=ax,
                       color_threshold=0.7 * max(Z[:, 2]))

    for lbl in ax.get_xticklabels():
        spk = lbl.get_text()
        row = meta_df[meta_df["spk"] == spk]
        if not row.empty:
            l1  = row.iloc[0]["L1"]
            gen = row.iloc[0]["Gender"]
            lbl.set_color(l1_color.get(l1, "black"))
            lbl.set_text(f"{spk}\n({gen})")

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Ward distance", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    handles = [Patch(color=l1_color[v], label=v) for v in l1_uniq]
    ax.legend(handles=handles, title="L1", fontsize=8, loc="upper right")

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
    ax.set_title("Silhouette — speaker clustering", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath}")


def plot_ari_heatmap(ari_df, outpath):
    fig, ax = plt.subplots(figsize=(max(5, len(ari_df.columns) * 1.5),
                                    max(3, len(ari_df) * 0.6)))
    im = ax.imshow(ari_df.values.astype(float),
                   vmin=-0.3, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(ari_df.columns)))
    ax.set_xticklabels(ari_df.columns, fontsize=10)
    ax.set_yticks(range(len(ari_df.index)))
    ax.set_yticklabels(ari_df.index, fontsize=9)
    for i in range(len(ari_df.index)):
        for j in range(len(ari_df.columns)):
            v = float(ari_df.values[i, j])
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
    plt.colorbar(im, ax=ax, label="ARI")
    ax.set_title("ARI — speaker clustering vs L1 and gender", fontsize=10)
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
    parser.add_argument("--k-values", default="2,3,4")
    parser.add_argument("--outdir",   default="./plots")
    parser.add_argument("--out-ari",  default="./tables/speaker_clustering_ari.csv")
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_ari), exist_ok=True)

    k_values = [int(k) for k in args.k_values.split(",")]

    # ── load & normalise ──────────────────────────────────────────────────────
    df = pd.read_csv(args.acoustic)
    df["phoneme"] = df["phoneme"].fillna("")
    df = lobanov_norm(df)
    df_v = df[df["phoneme"].apply(is_vowel)].copy().reset_index(drop=True)

    vowel_pos = df[df["phoneme"].apply(is_vowel)].index.to_numpy()
    vowels    = sorted(df_v["phoneme"].unique())
    print(f"Vowels  : {vowels}")
    print(f"Speakers: {df_v['spk'].nunique()}")
    print(f"L1 groups: {df_v['L1'].unique().tolist()}")
    print(f"Gender groups: {df_v['Gender'].unique().tolist()}\n")

    def load_reps(path):
        data = np.load(path, allow_pickle=True)
        reps = data["word_reps"]
        if reps.dtype == object:
            reps = np.stack(reps, axis=0)
        return reps.astype(float)[vowel_pos]

    reps_wh = load_reps(args.whisper)
    reps_xl = load_reps(args.xlsr)

    # ── build speaker vectors ─────────────────────────────────────────────────
    meta_ac, X_ac = speaker_acoustic_vectors(df_v, vowels)
    meta_wh, X_wh = speaker_neural_vectors(df_v, reps_wh, vowels)
    meta_xl, X_xl = speaker_neural_vectors(df_v, reps_xl, vowels)

    # impute + scale acoustic
    X_ac = impute_colmean(X_ac)
    X_ac = StandardScaler().fit_transform(X_ac)

    # impute neural (rare missing vowels for a speaker)
    X_wh = impute_colmean(X_wh)
    X_xl = impute_colmean(X_xl)

    representations = {
        "acoustic":  (X_ac, meta_ac, "euclidean"),
        "whisper":   (X_wh, meta_wh, "cosine"),
        "xlsr":      (X_xl, meta_xl, "cosine"),
    }

    all_results = {}
    ari_rows    = []

    for rep_name, (X, meta_df, metric) in representations.items():
        print(f"── {rep_name} ({metric}) ────────────────────────────────────")
        D = euclidean_dist_matrix(X) if metric == "euclidean" \
            else cosine_dist_matrix(X)

        speakers = meta_df["spk"].tolist()
        res = cluster_and_evaluate(D, meta_df, k_values)
        all_results[rep_name] = (res, meta_df)

        # dendrogram
        plot_dendrogram(
            res["linkage"], speakers, meta_df,
            title=f"Speaker dendrogram — {rep_name}",
            outpath=f"{args.outdir}/dendrogram_speakers_{rep_name}.png",
        )

        # print table
        print(f"  {'k':>3}  {'silhouette':>10}  {'ARI_L1':>8}  {'ARI_Gender':>10}")
        for k in k_values:
            sil = res["silhouette"].get(k, np.nan)
            a_l1  = res["ari"][k].get("L1",     np.nan)
            a_gen = res["ari"][k].get("Gender",  np.nan)
            print(f"  {k:>3}  {sil:>10.4f}  {a_l1:>8.4f}  {a_gen:>10.4f}")

            ari_rows.append({
                "representation": rep_name,
                "k":              k,
                "silhouette":     sil,
                "ARI_L1":         a_l1,
                "ARI_Gender":     a_gen,
            })
        print()

    # ── silhouette plot ───────────────────────────────────────────────────────
    plot_silhouette(
        {n: r for n, (r, _) in all_results.items()},
        k_values,
        outpath=f"{args.outdir}/silhouette_speakers.png"
    )

    # ── ARI table + heatmap ───────────────────────────────────────────────────
    ari_df = pd.DataFrame(ari_rows)
    ari_df.to_csv(args.out_ari, index=False)
    print(f"Saved → {args.out_ari}")

    # best-k heatmap
    best_rows = []
    for rep_name, (res, meta_df) in all_results.items():
        sub    = ari_df[ari_df["representation"] == rep_name]
        best_k = int(sub.loc[sub["silhouette"].idxmax(), "k"])
        best   = sub[sub["k"] == best_k].iloc[0]
        best_rows.append({
            "representation": rep_name,
            "best_k":   best_k,
            "ARI_L1":   round(float(best["ARI_L1"]),    4),
            "ARI_Gender": round(float(best["ARI_Gender"]), 4),
        })
        print(f"{rep_name}: best k={best_k}  "
              f"ARI_L1={best['ARI_L1']:.4f}  "
              f"ARI_Gender={best['ARI_Gender']:.4f}")

    heatmap_df = pd.DataFrame(best_rows).set_index("representation")[
        ["ARI_L1", "ARI_Gender"]
    ]
    plot_ari_heatmap(
        heatmap_df,
        outpath=f"{args.outdir}/ari_heatmap_speakers.png"
    )

    # ── cluster membership at k=2 (L1/L2 hypothesis) ─────────────────────────
    print("\n── Cluster membership at k=2 ────────────────────────────────────")
    for rep_name, (res, meta_df) in all_results.items():
        labels = fcluster(res["linkage"], 2, criterion="maxclust") - 1
        meta_df = meta_df.copy()
        meta_df["cluster"] = labels
        print(f"\n  {rep_name}:")
        print(meta_df[["spk", "L1", "Gender", "cluster"]].to_string(index=False))
        ari_l1  = adjusted_rand_score(meta_df["L1"].to_numpy(), labels)
        ari_gen = adjusted_rand_score(meta_df["Gender"].to_numpy(), labels)
        print(f"  ARI(L1)={ari_l1:.4f}  ARI(Gender)={ari_gen:.4f}")

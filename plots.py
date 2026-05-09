import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2
from normalise import is_vowel, lobanov_norm

def confidence_ellipse(x, y, ax, n_std=2.4477, **kwargs):
    # 95% ellipse -> sqrt(chi2.ppf(0.95, df=2)) ≈ 2.4477
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    from matplotlib.patches import Ellipse
    ellipse = Ellipse((np.mean(x), np.mean(y)),
                      width=width, height=height,
                      angle=theta, **kwargs)
    ax.add_patch(ellipse)

def make_group_label(df):
    return df["L1"].astype(str) + "/" + df["Gender"].astype(str)

def plot_vowel_chart(df, outdir):
    # === Plot 1: Vowel chart with centroids + 95% ellipses ===
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="F2_normed", y="F1_normed",
        hue="group", style="phoneme", alpha=0.25, ax=ax
    )

    # centroids + ellipses
    for (phon, grp), sub in df.groupby(["phoneme", "group"]):
        ax.scatter(sub["F2_normed"].mean(), sub["F1_normed"].mean(),
                   s=80, marker="X", color="black")
        confidence_ellipse(
            sub["F2_normed"].values,
            sub["F1_normed"].values,
            ax=ax, edgecolor="black", facecolor="none", linewidth=1
        )

    ax.set_title("Vowel chart (F1 vs F2) with 95% ellipses")
    ax.invert_yaxis()  # IPA convention
    ax.set_xlabel("F2 (Lobanov)")
    ax.set_ylabel("F1 (Lobanov)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{outdir}/vowel_chart.png", dpi=300)



def plot_boxplot(df, outdir):
    # === Plot 2: Boxplots of F1 / F2 by phoneme, stratified ===
    g1 = sns.catplot(
    data=df, x="phoneme", y="F1_normed",
    col="group", kind="box", col_wrap=2,
    height=4, aspect=1.2)
    
    g1.set_titles("Group: {col_name}")
    g1.set_axis_labels("Phoneme", "F1 (Lobanov)")
    g1.fig.suptitle("F1 by phoneme (group-stratified)", y=1.03)
    g1.savefig(f"{outdir}/boxplots_f1_by_group.png", dpi=300)

    g2 = sns.catplot(
        data=df, x="phoneme", y="F2_normed",
        col="group", kind="box", col_wrap=2,
        height=4, aspect=1.2
    )
    g2.set_titles("Group: {col_name}")
    g2.set_axis_labels("Phoneme", "F2 (Lobanov)")
    g2.fig.suptitle("F2 by phoneme (group-stratified)", y=1.03)
    g2.savefig(f"{outdir}/boxplots_f2_by_group.png", dpi=300)


def plot_intra_speaker_variability(df, outdir, subset):
    # === Plot 3: Intra-speaker variability (violin + strip) ===
    subset = [int(p.strip()) for p in subset.split(",") if p.strip()]
    sub = df[df["rep_id"].isin(subset)]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=sub, x="phoneme", y="F1_normed",
        hue="spk", inner=None, cut=0, ax=ax, linewidth=0.8
    )
    sns.stripplot(
        data=sub, x="phoneme", y="F1_normed",
        hue="spk", dodge=True, size=2, alpha=0.5, ax=ax
    )

    ax.set_title("Intra-speaker variability across repetitions (F1)")
    ax.set_ylabel("F1 (Lobanov)")
    ax.set_xlabel("Phoneme")
    ax.legend([], [], frameon=False)
    fig.tight_layout()
    fig.savefig(f"{outdir}/intra_speaker_variability.png", dpi=300)



def plot_2d_proj(df, outdir, by, neural_rep):
    reps = np.load(neural_rep)["word_reps"]
    reps = np.stack(reps, axis=0) if reps.dtype == object else reps
    df["neural_x"] = reps[:, 0]
    df["neural_y"] = reps[:, 1]

    if by == "phoneme":
        df = df[df["phoneme"].apply(is_vowel)]

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.scatterplot(
        data=df,
        x="neural_x",
        y="neural_y",
        hue=by,
        ax=ax,
    )
    ax.set_title(f"2D projection of {neural_rep} representations colored by " + by)
    # fig.tight_layout()
    out_suffix = neural_rep.split("/")[-1].replace(".npz", "").replace("features_", "")
    fig.savefig(f"{outdir}/2d_proj_{by.lower()}_{out_suffix}.png", dpi=300)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ac", default="./reps/features_acoustic.csv")
    parser.add_argument("--input-nr", default="./tables/metadata.csv")
    parser.add_argument("--outdir", default="./plots")
    parser.add_argument("--subset", default="1,2,3,4,5,6")
    parser.add_argument("--layer-wh", default=24, type=int)
    parser.add_argument("--layer-xlsr", default=24, type=int)
    # parser.add_argument("--model", default="whisper", choices=["whisper", "xlsr"])
    parser.add_argument("--dimred", default="pca2", choices=["pca2", "umap"])
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input_ac)
    df["phoneme"] = df["phoneme"].fillna("")
    df = lobanov_norm(df)

    df = df[df["phoneme"].apply(is_vowel)]
    df["group"] = make_group_label(df)


    plot_vowel_chart(df, args.outdir)
    plot_boxplot(df, args.outdir)
    plot_intra_speaker_variability(df, args.outdir, args.subset)

    df_nr = pd.read_csv(args.input_nr)
    df_nr["phoneme"] = df_nr["phoneme"].fillna("")

    whisper_rep = f"./reps/features_whisper_layer_{args.layer_wh}_{args.dimred}.npz"
    xlsr_rep = f"./reps/features_xlsr_layer_{args.layer_xlsr}_{args.dimred}.npz"
    
    plot_2d_proj(df_nr, args.outdir, by="phoneme", neural_rep=whisper_rep)
    plot_2d_proj(df_nr, args.outdir, by="L1", neural_rep=whisper_rep)
    plot_2d_proj(df_nr, args.outdir, by="Gender", neural_rep=whisper_rep)

    plot_2d_proj(df_nr, args.outdir, by="phoneme", neural_rep=xlsr_rep)
    plot_2d_proj(df_nr, args.outdir, by="L1", neural_rep=xlsr_rep)
    plot_2d_proj(df_nr, args.outdir, by="Gender", neural_rep=xlsr_rep)
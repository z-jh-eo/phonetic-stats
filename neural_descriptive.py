import argparse
import numpy as np
import pandas as pd


def between_class_ratio(df, x_col="x", y_col="y", label_col="phoneme"):
    # drop empty phoneme rows
    df = df[df[label_col].astype(str).str.len() > 0].copy()

    X = df[[x_col, y_col]].to_numpy()
    mu = X.mean(axis=0)

    total = ((X - mu) ** 2).sum()

    between = 0.0
    for phon, sub in df.groupby(label_col):
        Xk = sub[[x_col, y_col]].to_numpy()
        nk = len(Xk)
        muk = Xk.mean(axis=0)
        between += nk * ((muk - mu) ** 2).sum()

    return between / total if total > 0 else np.nan


def cosine_similarity_matrix(X):
    # X: (n, d)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


def within_between_cosine_ratio(df, X, label_col="phoneme"):
    # drop empty phoneme rows
    keep = df[label_col].astype(str).str.len() > 0
    df = df[keep].reset_index(drop=True)
    X = X[keep.to_numpy()]

    S = cosine_similarity_matrix(X)
    labels = df[label_col].to_numpy()
    n = len(labels)

    # mask out diagonal
    mask = ~np.eye(n, dtype=bool)

    same = (labels[:, None] == labels[None, :]) & mask
    diff = (labels[:, None] != labels[None, :])

    within = S[same].mean() if same.any() else np.nan
    between = S[diff].mean() if diff.any() else np.nan
    ratio = within / between if between and not np.isnan(between) else np.nan
    return within, between, ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default="./tables/metadata.csv")
    parser.add_argument("--outdir", type=str, default="./tables/between_class_ratio.csv")
    parser.add_argument("--whisper", type=str, required=True)
    parser.add_argument("--xlsr", type=str, required=True)
    args = parser.parse_args()

    reps = {
        "whisper": args.whisper,
        "xlsr": args.xlsr,
    }

    df_meta = pd.read_csv(args.metadata)
    res = []

    for name, rep in reps.items():
        rep_data = np.load(rep)["word_reps"]
        # ensure 2D array
        rep_data = np.stack(rep_data) if rep_data.dtype == object else rep_data

        df = df_meta.copy()
        df["x"] = rep_data[:, 0]
        df["y"] = rep_data[:, 1]

        bcr = between_class_ratio(df, x_col="x", y_col="y", label_col="phoneme")
        within, between, cos_ratio = within_between_cosine_ratio(df, rep_data, label_col="phoneme")

        res.append({
            "model": name,
            "between_class_ratio": bcr,
            "within_cosine": within,
            "between_cosine": between,
            "within_between_cosine_ratio": cos_ratio,
        })

    res_df = pd.DataFrame(res)
    res_df.to_csv(args.outdir, index=False)

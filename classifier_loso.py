import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from normalise import is_vowel, lobanov_norm


# ── nearest-centroid classifier ───────────────────────────────────────────────

def nearest_centroid_predict(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray,
                              metric: str = "euclidean") -> np.ndarray:
    """
    Fit centroids on train, predict by nearest centroid on test.
    metric: 'euclidean' for acoustic, 'cosine' for neural.
    """
    classes = np.unique(y_train)
    centroids = np.array([X_train[y_train == c].mean(axis=0) for c in classes])

    if metric == "cosine":
        # normalise both test points and centroids
        def norm(M):
            n = np.linalg.norm(M, axis=1, keepdims=True)
            return M / np.where(n == 0, 1e-10, n)
        sims = norm(X_test) @ norm(centroids).T   # (n_test, n_classes)
        pred_idx = sims.argmax(axis=1)
    else:
        # euclidean: argmin distance
        diffs = X_test[:, None, :] - centroids[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        pred_idx = dists.argmin(axis=1)

    return classes[pred_idx]


def loso_cv(df: pd.DataFrame, X: np.ndarray,
            metric: str = "euclidean") -> tuple[np.ndarray, np.ndarray]:
    """
    Leave-one-speaker-out cross-validation.
    Returns (y_true, y_pred) aligned arrays over all tokens.
    """
    speakers = df["spk"].unique()
    y_true_all = []
    y_pred_all = []

    for spk in speakers:
        test_mask  = df["spk"] == spk
        train_mask = ~test_mask

        X_train = X[train_mask]
        y_train = df.loc[train_mask, "phoneme"].to_numpy()
        X_test  = X[test_mask]
        y_test  = df.loc[test_mask,  "phoneme"].to_numpy()

        # skip if any class is absent from training set
        if len(np.unique(y_train)) < 2:
            continue

        # drop NaN rows in train
        valid_train = ~np.isnan(X_train).any(axis=1)
        valid_test  = ~np.isnan(X_test).any(axis=1)
        if valid_train.sum() < 2 or valid_test.sum() == 0:
            continue

        preds = nearest_centroid_predict(
            X_train[valid_train], y_train[valid_train],
            X_test[valid_test], metric=metric
        )
        y_true_all.extend(y_test[valid_test])
        y_pred_all.extend(preds)

    return np.array(y_true_all), np.array(y_pred_all)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             phonemes: list) -> dict:
    acc = float((y_true == y_pred).mean())
    f1_per = f1_score(y_true, y_pred, labels=phonemes,
                      average=None, zero_division=0)
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=phonemes)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_per_class": dict(zip(phonemes, f1_per.tolist())),
        "confusion_matrix": cm,
    }


def evaluate_by_group(df: pd.DataFrame, y_true: np.ndarray,
                      y_pred: np.ndarray, phonemes: list,
                      l1_col: str = "L1") -> dict:
    """Accuracy and macro-F1 split by L1/L2 group."""
    results = {}
    groups = df["L1"].unique()
    # we need the group label for each token in the LOSO output
    # df is already filtered to vowels and reset; y_true aligns with it
    # rebuild group array matching y_true order via LOSO (same speaker loop order)
    # easier: recompute from df directly since LOSO preserves speaker order
    # we pass the group series aligned to the same token order as y_true
    for g in groups:
        mask = df_group_mask(df, g, y_true)  # defined below
        if mask.sum() == 0:
            continue
        acc = float((y_true[mask] == y_pred[mask]).mean())
        f1  = float(f1_score(y_true[mask], y_pred[mask],
                             average="macro", zero_division=0))
        results[g] = {"accuracy": acc, "f1_macro": f1, "n": int(mask.sum())}
    return results


def df_group_mask(df: pd.DataFrame, group: str,
                  y_true: np.ndarray) -> np.ndarray:
    """
    Reconstruct which positions in y_true belong to `group`.
    LOSO iterates speakers in df["spk"].unique() order, collecting
    test tokens; we reproduce the same order to build the mask.
    """
    speakers = df["spk"].unique()
    mask = []
    for spk in speakers:
        test_mask = (df["spk"] == spk).to_numpy()
        valid_test = ~np.isnan(np.zeros((test_mask.sum(), 1))).any(axis=1)
        n = test_mask.sum()
        g_vals = df.loc[df["spk"] == spk, "L1"].to_numpy()
        mask.extend((g_vals == group).tolist())
    return np.array(mask[:len(y_true)])


# ── McNemar test ──────────────────────────────────────────────────────────────

def mcnemar_test(y_true: np.ndarray,
                 y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> dict:
    """
    McNemar test on matched pairs between two classifiers.
    Compares classifier A vs classifier B on the same token set.
    """
    correct_a = (y_true == y_pred_a)
    correct_b = (y_true == y_pred_b)

    # contingency table:
    # [both correct,  A correct B wrong]
    # [A wrong B correct, both wrong  ]
    n11 = int(( correct_a &  correct_b).sum())
    n10 = int(( correct_a & ~correct_b).sum())
    n01 = int((~correct_a &  correct_b).sum())
    n00 = int((~correct_a & ~correct_b).sum())

    table = np.array([[n11, n10], [n01, n00]])
    # use exact=True (binomial) when discordant pairs < 25, else chi2
    discordant = n10 + n01
    exact = discordant < 25
    result = mcnemar(table, exact=exact, correction=not exact)

    return {
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "discordant": discordant,
        "statistic": round(float(result.statistic), 4),
        "p_value":   round(float(result.pvalue), 6),
        "exact":     exact,
    }


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic", default="./reps/features_acoustic.csv")
    parser.add_argument("--whisper",  default="./reps/features_whisper.npz")
    parser.add_argument("--xlsr",     default="./reps/features_xlsr.npz")
    parser.add_argument("--out-scores",  default="./tables/classifier_scores.csv")
    parser.add_argument("--out-f1",      default="./tables/classifier_f1_per_class.csv")
    parser.add_argument("--out-cm",      default="./tables/classifier_confusion_matrices.csv")
    parser.add_argument("--out-mcnemar", default="./tables/classifier_mcnemar.csv")
    args = parser.parse_args()

    # ── load & normalise ──────────────────────────────────────────────────────
    df = pd.read_csv(args.acoustic)
    df["phoneme"] = df["phoneme"].fillna("")
    df = lobanov_norm(df)

    df_v = df[df["phoneme"].apply(is_vowel)].copy().reset_index(drop=True)
    vowel_pos = df[df["phoneme"].apply(is_vowel)].index.to_numpy()
    vowels = sorted(df_v["phoneme"].unique())
    print(f"Vowels: {vowels}")
    print(f"Tokens: {len(df_v)}  |  Speakers: {df_v['spk'].nunique()}\n")

    # ── feature matrices ──────────────────────────────────────────────────────
    X_ac = df_v[["F1_normed", "F2_normed"]].to_numpy(dtype=float)

    def load_reps(path):
        data = np.load(path, allow_pickle=True)
        reps = data["word_reps"]
        if reps.dtype == object:
            reps = np.stack(reps, axis=0)
        return reps.astype(float)[vowel_pos]

    X_wh = load_reps(args.whisper)
    X_xl = load_reps(args.xlsr)

    representations = {
        "acoustic": (X_ac, "euclidean"),
        "whisper":  (X_wh, "cosine"),
        "xlsr":     (X_xl, "cosine"),
    }

    # ── LOSO CV ───────────────────────────────────────────────────────────────
    predictions = {}
    scores = []
    f1_rows = []
    cm_rows = []

    for name, (X, metric) in representations.items():
        print(f"Running LOSO CV — {name} ({metric}) …")
        y_true, y_pred = loso_cv(df_v, X, metric=metric)
        predictions[name] = (y_true, y_pred)

        ev = evaluate(y_true, y_pred, vowels)
        print(f"  accuracy={ev['accuracy']:.4f}  macro-F1={ev['f1_macro']:.4f}")

        scores.append({
            "model": name,
            "accuracy": round(ev["accuracy"], 4),
            "f1_macro": round(ev["f1_macro"], 4),
            "n_tokens": len(y_true),
        })

        for ph, f1v in ev["f1_per_class"].items():
            f1_rows.append({"model": name, "phoneme": ph,
                            "f1": round(f1v, 4)})

        cm = ev["confusion_matrix"]
        for i, pi in enumerate(vowels):
            for j, pj in enumerate(vowels):
                cm_rows.append({"model": name,
                                "true": pi, "pred": pj,
                                "count": int(cm[i, j])})

    # ── save scores ───────────────────────────────────────────────────────────
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(args.out_scores, index=False)
    print(f"\nOverall scores:\n{scores_df.to_string(index=False)}")

    f1_df = pd.DataFrame(f1_rows)
    f1_wide = f1_df.pivot(index="phoneme", columns="model", values="f1")
    f1_wide.to_csv(args.out_f1)
    print(f"\nPer-class F1:\n{f1_wide.to_string()}")

    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(args.out_cm, index=False)

    # ── McNemar tests (all pairs) ─────────────────────────────────────────────
    print("\nMcNemar tests …")
    names = list(representations.keys())
    mcnemar_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            yt_a, yp_a = predictions[na]
            yt_b, yp_b = predictions[nb]
            # align on common token set (should be identical if no NaN mismatch)
            min_n = min(len(yt_a), len(yt_b))
            res = mcnemar_test(yt_a[:min_n], yp_a[:min_n], yp_b[:min_n])
            res["model_A"] = na
            res["model_B"] = nb
            mcnemar_rows.append(res)
            sig = "**" if res["p_value"] < 0.05 else ""
            print(f"  {na} vs {nb}: stat={res['statistic']:.4f} "
                  f"p={res['p_value']:.4f} {sig}")

    mc_df = pd.DataFrame(mcnemar_rows)
    col_order = ["model_A", "model_B", "n11", "n10", "n01", "n00",
                 "discordant", "statistic", "p_value", "exact"]
    mc_df = mc_df[col_order]
    mc_df.to_csv(args.out_mcnemar, index=False)

    # ── per-group accuracy ────────────────────────────────────────────────────
    def get_group_labels(df_v):
        groups = []
        for spk in df_v["spk"].unique():
            test_mask = df_v["spk"] == spk
            groups.extend(df_v.loc[test_mask, "L1"].tolist())
        return np.array(groups)

    group_labels = get_group_labels(df_v)

    print("\n── Per-group accuracy ───────────────────────────────────────────")
    group_rows = []
    for name in representations:
        yt, yp = predictions[name]
        gl = group_labels[:len(yt)]
        for g in sorted(set(gl)):
            mask = gl == g
            acc = float((yt[mask] == yp[mask]).mean())
            f1  = float(f1_score(yt[mask], yp[mask], average="macro", zero_division=0))
            group_rows.append({"model": name, "group": g,
                               "accuracy": round(acc, 4), "f1_macro": round(f1, 4),
                               "n": int(mask.sum())})
            print(f"  {name} / {g}: acc={acc:.4f}  F1={f1:.4f}  n={mask.sum()}")

    group_df = pd.DataFrame(group_rows)
    group_df.to_csv("tables/classifier_scores_by_group.csv", index=False)

    print(f"\nSaved → {args.out_scores}")
    print(f"Saved → {args.out_f1}")
    print(f"Saved → {args.out_cm}")
    print(f"Saved → {args.out_mcnemar}")

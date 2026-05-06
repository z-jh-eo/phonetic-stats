import argparse
import numpy as np
import umap
from sklearn.decomposition import PCA


def load_reps(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    reps = data["word_reps"]
    if reps.dtype == object:
        reps = np.stack(reps, axis=0)
    return reps


def reduce_dim(reps: np.ndarray, method: str, d: int, seed: int) -> np.ndarray:
    if method == "pca":
        reducer = PCA(n_components=d, random_state=seed)
        return reducer.fit_transform(reps)
    elif method == "umap":
        reducer = umap.UMAP(
            n_components=d,
            random_state=seed,
            metric="cosine"
        )
        return reducer.fit_transform(reps)
    else:
        raise ValueError("method must be 'pca' or 'umap'")


def process(input_path: str, output_path: str, method: str, d: int, seed: int):
    reps = load_reps(input_path)

    # mask valid rows (no NaNs)
    mask = ~np.isnan(reps).any(axis=1)

    # reduce only valid rows
    reduced_valid = reduce_dim(reps[mask], method, d, seed)

    # re-expand to full length (preserve blank rows)
    reduced_full = np.full((reps.shape[0], d), np.nan, dtype=np.float32)
    reduced_full[mask] = reduced_valid

    np.savez(output_path, word_reps=reduced_full, keep_mask=mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["pca", "umap"], default="pca")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whisper", default="./reps/features_whisper.npz")
    parser.add_argument("--xlsr", default="./reps/features_xlsr.npz")
    args = parser.parse_args()

    out_suffix = f"{args.method}{args.dim}"

    process(args.whisper,
            f"./reps/features_whisper_{out_suffix}.npz",
            args.method, args.dim, args.seed)

    process(args.xlsr,
            f"./reps/features_xlsr_{out_suffix}.npz",
            args.method, args.dim, args.seed)

    print("Done.")
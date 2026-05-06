import argparse
import pandas as pd
from normalise import is_vowel, lobanov_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./reps/features_acoustic.csv")
    parser.add_argument("--output", type=str, default="./tables/descriptive_report.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["phoneme"]=df["phoneme"].fillna("")

    df = lobanov_norm(df)

    res_1 = df[df["phoneme"].apply(is_vowel)]\
            .groupby(["L1", "Gender", "phoneme"])[["F1_normed", "F2_normed"]]\
            .agg(["mean", "median", "std"])
    
    q1 = df[df["phoneme"].apply(is_vowel)]\
            .groupby(["L1", "Gender", "phoneme"])[["F1_normed", "F2_normed"]]\
            .quantile(0.25)
    
    q3 = df[df["phoneme"].apply(is_vowel)]\
            .groupby(["L1", "Gender", "phoneme"])[["F1_normed", "F2_normed"]]\
            .quantile(0.75)
    
    iqr = q3 - q1

    res_1[("F1_normed", "cv")] = res_1[("F1_normed", "std")] / res_1[("F1_normed", "mean")]
    res_1[("F2_normed", "cv")] = res_1[("F2_normed", "std")] / res_1[("F2_normed", "mean")]

    res_1[("F1_normed", "iqr")] = iqr["F1_normed"]
    res_1[("F2_normed", "iqr")] = iqr["F2_normed"]
    res_1 = res_1.sort_index(axis=1)

    res_1.to_csv(args.output, index=False)

    
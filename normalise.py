import pandas as pd


VOWELS = {"a", "ɑ", "e", "ɛ", "i", "ɪ", "o", "ø", "œ", "u", "y"}

def is_vowel(phoneme: str) -> bool:
    return phoneme.lower().strip() in VOWELS


def lobanov_norm(df: pd.DataFrame):
    vowel_df = df[df["phoneme"].apply(is_vowel)]

    stats = vowel_df.groupby("spk")[["F1", "F2"]].agg(["mean", "std"])
    stats.columns = ["_".join(col) for col in stats.columns]

    df = df.copy()
    df["F1_normed"] = (df["F1"] - df["spk"].map(stats["F1_mean"])) / df["spk"].map(stats["F1_std"])
    df["F2_normed"] = (df["F2"] - df["spk"].map(stats["F2_mean"])) / df["spk"].map(stats["F2_std"])
    return df


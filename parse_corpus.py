import textgrid
import os
import re
import pandas as pd


DATA_PATH = "./ru-fr_interference/2/wav_et_textgrids/FRcorp_textgrids_only"
CORR_PATH = "./ru-fr_interference/2/RUFRcorr.csv"
META_PATH = "./ru-fr_interference/2/metadata_RUFR.csv"

def read_textgrid(in_path: str):
    phone_rows = []
    with os.scandir(in_path) as d:
        for f in d:
            if f.is_file() and f.name.endswith("TextGrid"):
                wav_path = f"{in_path}/{f.name.replace('.TextGrid', '.wav')}"
                f_split = f.name.split(".")[0].split("_")
                spk_id = f_split[0].upper()
                sent_id = int(re.search(r"\d+", f_split[3]).group(0))

                tg = textgrid.TextGrid.fromFile(f"{in_path}/{f.name}")
                for phone in tg[1]: # through intervals (phones) of the 2nd tier 
                    label = phone.mark
                    onset = phone.minTime
                    offset = phone.maxTime
                    dur = phone.maxTime - phone.minTime
                    phone_rows.append({
                        "spk": spk_id,
                        "sent_id": sent_id,
                        "phoneme": label,
                        "onset": onset,
                        "offset": offset,
                        "dur": dur,
                        "wav_path": wav_path,
                    })
    return phone_rows


if __name__ == "__main__":

    all_phone_rows = []
    with os.scandir(DATA_PATH) as d:
        for spk in d:
            if spk.is_dir():
                phone_rows = read_textgrid(f"{DATA_PATH}/{spk.name}")
                all_phone_rows.extend(phone_rows)
    
    phone_df = pd.DataFrame(all_phone_rows)
    meta_df = pd.read_csv(META_PATH, sep=";")[["spk", "L1", "Age", "Gender"]]

    df = pd.merge(meta_df, phone_df, on="spk", how="right")\
           .sort_values(by=["spk", "sent_id", "onset"])\
           .reset_index(drop=True)

    corr_df = pd.read_csv(CORR_PATH, sep="\t")
    corr_df = corr_df.melt(id_vars=["Word"],
                           value_vars=["occ.1", "occ.2", "occ.3", "occ.4", "occ.5", "occ.6"],
                           var_name="rep_id",
                           value_name="sent_id")
    corr_df["rep_id"] = corr_df["rep_id"].str.replace("occ.", "").astype(int)
    corr_df["sent_id"] = corr_df["sent_id"].astype(int)

    corr_df = corr_df[["rep_id", "sent_id"]]
    df = pd.merge(df, corr_df, on="sent_id", how="left")

    os.makedirs("./tables", exist_ok=True)
    df.to_csv("./tables/metadata.csv", index=False)






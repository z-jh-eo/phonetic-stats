import argparse
import torch
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor


def batch_iter(series: pd.Series, batch_size: int = 8):
    for i in range(0, len(series), batch_size):
        yield series.iloc[i:i+batch_size]


def batch_proc(batch_paths: pd.Series, which_layer: int) -> tuple[list[torch.Tensor], list[int]]:
    audio_list = []
    sample_nbs = []
    for p in batch_paths:
        wav, sr = sf.read(p)
        if sr != 16_000:
            raise ValueError("SR need to be set to 16k")
        sample_nbs.append(len(wav))
        audio_list.append(wav)

    inputs = processor(audio_list, sampling_rate=16_000,
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    frame_lengths = model._get_feat_extract_output_lengths(
        torch.tensor([len(x) for x in audio_list])
    )

    trimmed: list[torch.Tensor] = []
    for i, L in enumerate(frame_lengths):
        L = int(L.item())
        trimmed.append(out.hidden_states[which_layer][i, :L].cpu())

    return trimmed, sample_nbs


def word_to_frames(start_s: float, end_s: float, n_samples: int, sr: int = 16_000) -> tuple[int, int]:
    n_frames = model._get_feat_extract_output_lengths(
        torch.tensor([n_samples])
    ).item()

    duration_s = n_samples / sr

    start_f = int(start_s * n_frames / duration_s)
    end_f = max(start_f + 1, int(end_s * n_frames / duration_s))
    end_f = min(end_f, n_frames)
    return start_f, end_f


def mean_pool_word(frame_reps: torch.Tensor, start_s: float,
                   end_s: float, n_samples: int) -> torch.Tensor:
    start_f, end_f = word_to_frames(start_s, end_s, n_samples)
    return frame_reps[start_f:end_f].mean(dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dtype", "-d", default=torch.float64)
    parser.add_argument("--metadata-path", "-m", default="./metadata.csv")
    #parser.add_argument("--output-path", "-o", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--which-layer", "-l", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model {args.model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2Model.from_pretrained(args.model_name)
    model.eval()
    model.to(device)

    df = pd.read_csv(args.metadata_path)

    audio = pd.Series(df["wav_path"].unique())

    all_frame_reps: list[torch.Tensor] = []
    all_sample_nbs: list[int] = []

    print("Computing frame representations...")
    for paths in batch_iter(audio):
        frame_reps, sample_nbs = batch_proc(paths, which_layer=args.which_layer)
        all_frame_reps.extend(frame_reps)
        all_sample_nbs.extend(sample_nbs)

    audio_df = pd.DataFrame({"wav_path": audio.values,
                             "frame_reps": all_frame_reps,
                             "n_samples": all_sample_nbs})

    df_merged = df.merge(audio_df, how="left", on="wav_path")

    word_reps: list[torch.Tensor] = []
    for row in tqdm(df_merged.itertuples(index=False), total=df_merged.shape[0], 
                    desc="Aggregating word representations..."):
        # for row in df_merged.itertuples(index=False)
        rep = mean_pool_word(row.frame_reps, row.onset, row.offset, row.n_samples)
        word_reps.append(rep)

    df_merged.drop(["frame_reps", "n_samples"], axis=1, inplace=True)

    df_merged.to_csv("metadata_with_reps.csv", index=False)
    torch.save(torch.stack(word_reps),
               f"reps_layer{args.which_layer}_{args.model_name.replace('/', '_')}.pt")
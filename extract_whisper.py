import argparse
import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from transformers import WhisperModel, AutoFeatureExtractor


# Whisper's CNN encoder always outputs exactly 1500 frames for a 30-s window
WHISPER_N_ENCODER_FRAMES = 1500
WHISPER_SECONDS          = 30.0   # the fixed window the encoder sees


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

    inputs = feature_extractor(
        audio_list,
        sampling_rate=16_000,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    decoder_input_ids = torch.full(
        (inputs["input_features"].size(0), 1),
        model.config.decoder_start_token_id,
        device=device,
        dtype=torch.long,
    )

    with torch.no_grad():
        out = model(
            **inputs,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True
        )

    trimmed: list[torch.Tensor] = []
    for i, n_samp in enumerate(sample_nbs):
        actual_duration = min(n_samp / 16_000, WHISPER_SECONDS)
        valide_frames = int(actual_duration / WHISPER_SECONDS * WHISPER_N_ENCODER_FRAMES)
        valide_frames = max(valide_frames, 1)
        trimmed.append(out.encoder_hidden_states[which_layer][i, :valide_frames].cpu())

    return trimmed, sample_nbs


def word_to_frames(start_s: float, end_s: float, n_samples: int, sr: int = 16_000) -> tuple[int, int]:

    duration_s = n_samples / sr
    duration_s = min(duration_s, WHISPER_SECONDS)

    frames_per_sec = WHISPER_N_ENCODER_FRAMES / WHISPER_SECONDS
    start_f = int(start_s * frames_per_sec)
    end_f = max(start_f + 1, int(end_s * frames_per_sec))
    end_f = min(end_f, int(duration_s * frames_per_sec))
    return start_f, end_f


def mean_pool_word(frame_reps: torch.Tensor, start_s: float,
                   end_s: float, n_samples: int) -> torch.Tensor:
    start_f, end_f = word_to_frames(start_s, end_s, n_samples)
    return frame_reps[start_f:end_f].mean(dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dtype", "-d", default=torch.float64)
    parser.add_argument("--metadata-path", "-m", default="./tables/metadata.csv")
    #parser.add_argument("--output-path", "-o", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="openai/whisper-medium")
    parser.add_argument("--which-layer", "-l", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model {args.model_name}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = WhisperModel.from_pretrained(args.model_name)
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

    # df_merged.drop(["frame_reps", "n_samples"], axis=1, inplace=True)

    # df_merged.to_csv("./tables/metadata_with_reps.csv", index=False)
    os.makedirs("./reps", exist_ok=True)
    word_reps_np = [rep.numpy() for rep in word_reps]
    np.savez("./reps/features_whisper.npz", word_reps=word_reps_np)

    # torch.save(torch.stack(word_reps),f"reps_layer{args.which_layer}_{args.model_name.replace('/', '_')}.pt")
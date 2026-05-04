import argparse
import librosa
import os
import soundfile as sf
from pathlib import Path


def resample_to_16k(in_path: str, out_path: str | None = None) -> None:
    y, sr = librosa.load(in_path, sr=None)      # load at original sr
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    if out_path is None:
        out_path = in_path
    sf.write(out_path, y16, 16000)


def scan_dir(input_dir: str) -> list[str]:
    out: list[str] = []

    with os.scandir(input_dir) as d:
        for e in d:
            if e.is_dir():
                out.append(
                    input_dir+"/"+e.name
                )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str,
                        default="./wav_et_textgrids/FRcorp_textgrids_only/")
    args = parser.parse_args()

    speakers = scan_dir(args.input_dir)

    for s in speakers:
        with os.scandir(s) as d:
            for e in d:
                if e.is_file and e.name.endswith("wav"):
                    resample_to_16k(s+"/"+e.name)

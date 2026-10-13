"""Run MediaPipe on every video in the manifest and cache landmarks as .npy files.

Usage:
    python -m signlang.extract_landmarks --config config.yaml
Safe to re-run: videos that already have a .npy file are skipped.
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from signlang.config import load_config
from signlang.features import hand_presence
from signlang.landmarks import LandmarkExtractor


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()


def extract_video(extractor, path, frame_start=1, frame_end=-1):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start = max(int(frame_start) - 1, 0)
    end = None if int(frame_end) <= 0 else int(frame_end)
    extractor.reset()
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (end is not None and i >= end):
            break
        if i >= start:
            frames.append(extractor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1000.0 / fps))
        i += 1
    cap.release()
    return np.stack(frames).astype(np.float32) if frames else np.zeros((0, 144), np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    manifest = pd.read_csv(cfg["paths"]["video_manifest"])
    out_dir = Path(cfg["paths"]["landmarks_dir"])
    extractor = LandmarkExtractor(cfg["paths"]["mediapipe_models"])

    rows, dropped = [], 0
    for n, r in enumerate(manifest.itertuples(index=False), 1):
        npy = out_dir / slug(r.label) / f"{Path(r.video_path).stem}.npy"
        npy.parent.mkdir(parents=True, exist_ok=True)
        if npy.exists():
            seq = np.load(npy)
        else:
            seq = extract_video(extractor, r.video_path, r.frame_start, r.frame_end)
            if len(seq) >= cfg["min_frames"]:
                np.save(npy, seq)
        if len(seq) < cfg["min_frames"] or hand_presence(seq) < cfg["min_hand_fraction"]:
            dropped += 1
            npy.unlink(missing_ok=True)
        else:
            rows.append({"npy_path": str(npy), "label": r.label, "split": r.split, "signer_id": r.signer_id})
        if n % 25 == 0:
            print(f"{n}/{len(manifest)} videos processed")

    extractor.close()
    pd.DataFrame(rows).to_csv(cfg["paths"]["landmarks_csv"], index=False)
    print(f"Saved {len(rows)} samples to {cfg['paths']['landmarks_csv']} ({dropped} dropped: too short or hands rarely visible)")


if __name__ == "__main__":
    main()

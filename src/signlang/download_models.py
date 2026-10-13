"""Download the MediaPipe landmarker model files (one-time, ~15 MB).

Usage:
    python -m signlang.download_models
"""
import argparse
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models"
FILES = {
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/mediapipe")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = out / name
        if dest.exists():
            print(f"Already have {dest}")
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
    print("Done.")


if __name__ == "__main__":
    main()

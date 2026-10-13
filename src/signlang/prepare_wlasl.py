"""Build a video manifest from WLASL videos you have already downloaded.

WLASL ships a JSON file with video links, but many links are dead, so this script only keeps videos
that actually exist in --videos-dir (files named <video_id>.mp4) and picks the --top-k glosses with
the most available videos.

Usage:
    python -m signlang.prepare_wlasl --wlasl-json WLASL_v0.3.json --videos-dir videos --top-k 100
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def build_manifest(json_path, videos_dir, top_k=100, ext=".mp4") -> pd.DataFrame:
    entries = json.load(open(json_path, encoding="utf-8"))
    rows = []
    for entry in entries:
        for inst in entry["instances"]:
            path = Path(videos_dir) / f"{inst['video_id']}{ext}"
            if path.exists():
                rows.append({
                    "video_path": str(path),
                    "label": entry["gloss"],
                    "split": inst.get("split", ""),
                    "signer_id": inst.get("signer_id", -1),
                    "frame_start": inst.get("frame_start", 1),
                    "frame_end": inst.get("frame_end", -1),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No videos found in {videos_dir}. Check the folder and file names (<video_id>{ext}).")
    keep = df.groupby("label").size().sort_values(ascending=False).head(top_k).index
    return df[df["label"].isin(keep)].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wlasl-json", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--out", default="data/video_manifest.csv")
    args = parser.parse_args()

    df = build_manifest(args.wlasl_json, args.videos_dir, args.top_k)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} videos | {df['label'].nunique()} glosses | splits: {df['split'].value_counts().to_dict()}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

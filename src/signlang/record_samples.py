"""Record your own signs from the webcam (great for a custom vocabulary and a live demo).

Usage:
    python -m signlang.record_samples --label hello --n 30
Controls: SPACE = record one sample (after a short countdown), Q = quit.
Samples are saved straight to data/landmarks/<label>/ and appended to data/landmarks.csv.
Tip: vary your distance, lighting and speed a little between samples.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from signlang.config import load_config
from signlang.extract_landmarks import slug
from signlang.landmarks import LandmarkExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--label", required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg["paths"]["landmarks_dir"]) / slug(args.label)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(cfg["paths"]["landmarks_csv"])
    extractor = LandmarkExtractor(cfg["paths"]["mediapipe_models"])
    cap = cv2.VideoCapture(args.camera)
    existing = len(list(out_dir.glob("*.npy")))
    saved = 0
    print("Press SPACE to record, Q to quit.")

    while saved < args.n:
        ok, frame = cap.read()
        if not ok:
            break
        view = cv2.flip(frame, 1)
        cv2.putText(view, f"{args.label}: {existing + saved} samples (SPACE=record, Q=quit)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("record", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            for sec in (3, 2, 1):  # countdown so you can get into position
                ok, frame = cap.read()
                view = cv2.flip(frame, 1)
                cv2.putText(view, str(sec), (280, 260), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)
                cv2.imshow("record", view)
                cv2.waitKey(1000)
            extractor.reset()
            frames, t0, last = [], time.time(), time.time()
            while time.time() - t0 < args.seconds:
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.time()
                frames.append(extractor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (now - last) * 1000))
                last = now
                view = cv2.flip(frame, 1)
                cv2.putText(view, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.imshow("record", view)
                cv2.waitKey(1)
            if len(frames) >= cfg["min_frames"]:
                path = out_dir / f"{args.label}_{int(time.time())}.npy"
                np.save(path, np.stack(frames).astype(np.float32))
                row = pd.DataFrame([{"npy_path": str(path), "label": args.label, "split": "", "signer_id": "self"}])
                row.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
                saved += 1
                print(f"Saved sample {existing + saved}")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    main()

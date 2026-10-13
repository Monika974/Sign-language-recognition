"""Live webcam demo: shows the recognised sign as text.

Usage:
    python -m signlang.realtime --model-dir outputs/gru
Press Q to quit.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np

from signlang.config import load_config
from signlang.features import hand_presence
from signlang.inference import SignPredictor
from signlang.smoothing import PredictionSmoother
from signlang.landmarks import LandmarkExtractor


def draw_points(view, feats, mirror):
    h, w = view.shape[:2]
    pts = feats.reshape(-1, 3)
    for x, y, _ in pts[np.abs(pts).sum(axis=1) > 0]:
        x = 1 - x if mirror else x
        cv2.circle(view, (int(x * w), int(y * h)), 3, (0, 255, 255), -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-dir", required=True, help="e.g. outputs/gru")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-mirror", action="store_true", help="show the un-mirrored camera image")
    args = parser.parse_args()
    cfg = load_config(args.config)
    rt = cfg["realtime"]

    predictor = SignPredictor(args.model_dir)
    extractor = LandmarkExtractor(cfg["paths"]["mediapipe_models"])
    smoother = PredictionSmoother(rt["threshold"], rt["consensus"])
    buffer = deque(maxlen=rt["window"])
    mirror = not args.no_mirror

    cap = cv2.VideoCapture(args.camera)
    last, n, top = time.time(), 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        # Features come from the un-mirrored frame, exactly like the training videos.
        feats = extractor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (now - last) * 1000)
        last = now
        buffer.append(feats)
        n += 1

        if len(buffer) == rt["window"] and n % rt["stride"] == 0:
            seq = np.stack(buffer)
            if hand_presence(seq) >= rt["min_hand_fraction"]:
                probs = predictor.predict(seq)
                top = predictor.top_k(probs, 3)
                best = int(np.argmax(probs))
                smoother.update(best, float(probs[best]))
            else:
                top = []
                smoother.update(None)

        view = cv2.flip(frame, 1) if mirror else frame.copy()
        draw_points(view, feats, mirror)
        word = predictor.classes[smoother.current] if smoother.current is not None else "..."
        cv2.rectangle(view, (0, 0), (view.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(view, word, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        for i, (name, p) in enumerate(top):
            cv2.putText(view, f"{name} {p:.2f}", (10, 80 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Sign language recognition (Q to quit)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    main()

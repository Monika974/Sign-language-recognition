"""MediaPipe hand + pose landmark extraction (MediaPipe Tasks API).

The Tasks API needs two small model files (.task). Download them once with:
    python -m signlang.download_models
"""
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from signlang.features import LEFT, N_POINTS, POSE, POSE_IDS, RIGHT

HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker_lite.task"


class LandmarkExtractor:
    """Turns an RGB frame into a (144,) feature vector. Missing parts are zeros.

    Left/right slots follow MediaPipe's handedness label. Training and live inference both use
    un-mirrored frames, so the labels stay consistent between the two.
    """

    def __init__(self, model_dir: str | Path = "models/mediapipe"):
        self.model_dir = Path(model_dir)
        for name in (HAND_MODEL, POSE_MODEL):
            if not (self.model_dir / name).exists():
                raise FileNotFoundError(f"{self.model_dir / name} not found. Run: python -m signlang.download_models")
        self._build()

    def _build(self):
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_dir / HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_dir / POSE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
        )
        self.hands = vision.HandLandmarker.create_from_options(hand_opts)
        self.pose = vision.PoseLandmarker.create_from_options(pose_opts)
        self._ts = 0

    def reset(self):
        """Call between videos so tracking does not carry over."""
        self.close()
        self._build()

    def close(self):
        self.hands.close()
        self.pose.close()

    def process(self, frame_rgb: np.ndarray, dt_ms: float = 33.0) -> np.ndarray:
        self._ts += max(1, int(round(dt_ms)))  # MediaPipe needs strictly increasing timestamps
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        hand_res = self.hands.detect_for_video(image, self._ts)
        pose_res = self.pose.detect_for_video(image, self._ts)

        pts = np.zeros((N_POINTS, 3), dtype=np.float32)
        for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
            slot = LEFT if handed[0].category_name == "Left" else RIGHT
            pts[slot] = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
        if pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks[0]
            pts[POSE] = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in POSE_IDS], dtype=np.float32)
        return pts.reshape(-1)

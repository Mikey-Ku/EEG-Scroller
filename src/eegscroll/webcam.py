"""Webcam face tracking with MediaPipe: blink and head angles per frame."""

import threading
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from eegscroll.clock import now

# Face model, downloaded once (~4 MB)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"

# One row per webcam frame
COLUMNS = ["t", "face", "blink", "yaw", "pitch"]


def frame_row(t: float, result) -> dict:
    """One MediaPipe result -> one table row."""
    # No face found: mostly empty row
    if not result.face_blendshapes:
        return {"t": t, "face": 0}

    # Blink: mean eyes-closed score, 0 to 1
    scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
    blink = (scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2

    # Head angles from the face transform matrix
    rotation = np.asarray(result.facial_transformation_matrixes[0])[:3, :3]
    pitch, yaw, _roll = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    return {"t": t, "face": 1, "blink": blink, "yaw": yaw, "pitch": pitch}


class WebcamRecorder:
    """Background thread: camera frames -> face rows."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.rows: list[dict] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        # Download the face model once
        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading face model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        # Open camera; macOS may ask for permission
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error = "Could not open webcam (check camera permission)."
            print(self.error)
            return

        # Video mode tracks one face across frames
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while not self._stop.is_set():
                ok, frame = cap.read()
                t = now()
                if not ok:
                    continue

                # MediaPipe wants RGB and a millisecond timestamp
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.rows.append(frame_row(t, landmarker.detect_for_video(image, int(t * 1000))))
        cap.release()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=COLUMNS)

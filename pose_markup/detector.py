import cv2
import mediapipe as mp
import numpy as np

from pose_markup.converting_utils import convert_landmarks_to_keypoints


class PoseDetector:
    def __init__(self, model_path: str) -> None:
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        base_options = BaseOptions(model_asset_path=model_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.VIDEO
        )

        self.landmarker = PoseLandmarker.create_from_options(
            options)

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def predict_landmarks(
            self,
            frame: np.ndarray,
            timestamp_ms: int
    ) -> mp.tasks.vision.PoseLandmarkerResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def predict_keypoints(
            self,
            frame: np.ndarray,
            timestamp_ms: int
    ) -> np.ndarray:
        landmark_result = self.predict_landmarks(frame, timestamp_ms)
        return convert_landmarks_to_keypoints(landmark_result)

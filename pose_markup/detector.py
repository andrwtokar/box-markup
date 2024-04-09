import time
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
            base_options=base_options, running_mode=VisionRunningMode.VIDEO)

        self.landmarker = PoseLandmarker.create_from_options(options)
        self.total_prediction_time_ms = 0.0
        self.total_num_of_frames = 0

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def predict_landmarks(self, frame: np.ndarray, timestamp_ms: int) -> mp.tasks.vision.PoseLandmarkerResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def predict_keypoints(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        start = time.time()
        width, height = frame.shape[:2]
        landmark_result = self.predict_landmarks(frame, timestamp_ms)
        keypoints = convert_landmarks_to_keypoints(landmark_result)
        res = keypoints * [height, width, 1]
        res[:, :2] = res[:, :2].astype(np.int32).astype(np.float64)
        end = time.time()

        self.total_prediction_time_ms += (end - start) * 1000
        self.total_num_of_frames += 1

        return res

    def get_average_prediction_time(self):
        return round(self.total_prediction_time_ms / self.total_num_of_frames, 3)

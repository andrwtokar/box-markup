import mediapipe as mp
import numpy as np

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


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

        self.landmarker: mp.tasks.vision.PoseLandmarker = PoseLandmarker.create_from_options(
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
    ) -> mp.tasks.vision.PoseLandmarkerResul:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

    def predict_keypoints(
            self,
            frame: np.ndarray,
            timestamp_ms: int
    ) -> np.ndarray:
        landmark_result = self.predict_landmarks(frame, timestamp_ms)
        return PoseDetector.convert_landmarks_to_keypoints(landmark_result)

    @staticmethod
    def convert_landmarks_to_keypoints(
        pose_result: mp.tasks.vision.PoseLandmarkerResult
    ) -> np.ndarray:
        # TODO: Add filtering keypoints
        return [[landmark.x, landmark.y, landmark.visibility]
                for landmark in pose_result.pose_landmarks[0]]


def draw_landmarks(
        rgb_image: np.ndarray,
        detection_result: mp.tasks.vision.PoseLandmarker
) -> np.ndarray:
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )

    return annotated_image


def draw_keypoints(rgb_image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    pass


def draw_keypoints(
        rgb_image: np.ndarray,
        detection_result: mp.tasks.vision.PoseLandmarker
) -> np.ndarray:
    keypoints = PoseDetector.convert_landmarks_to_keypoints(detection_result)
    return draw_keypoints(rgb_image, keypoints)

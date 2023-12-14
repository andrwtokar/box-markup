import cv2
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
    ) -> mp.tasks.vision.PoseLandmarkerResult:
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
        landmarks = pose_result.pose_landmarks[0]
        __mask_17_points = {
            0: 0,  # nose
            1: 2,  # left eye
            2: 5,  # right eye
            3: 7,  # left ear
            4: 8,  # right ear
            5: 11,  # left shoulder
            6: 12,  # right shoulder
            7: 13,  # left elbow
            8: 14,  # right elbow
            9: 15,  # left wrist
            10: 16,  # right wrist
            11: 23,  # left hip
            12: 24,  # right hip
            13: 25,  # left knee
            14: 26,  # right knee
            15: 27,  # left ankle
            16: 28  # right ankle
        }

        mask_landmarks = [landmarks[i] for i in __mask_17_points.values()]
        return np.array([[landmark.x, landmark.y, landmark.visibility]
                         for landmark in mask_landmarks])


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


def unnormilized_keypoints(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    return (keypoints * [height, width, 1]).round()


def draw_pose(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    frame = draw_connections(frame, keypoints, thickness=2, alpha=0.7)
    frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)
    return frame


def draw_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    radius: int = 1,
    alpha: float = 1.0
) -> np.ndarray:
    __nose_color = (255, 255, 255)
    __left_color = (255, 69, 0)
    __right_color = (0, 191, 255)

    overlay = frame.copy()
    for index, keypoint in enumerate(keypoints):
        if index == 0:
            __draw_keypoint(overlay, keypoint, radius, __nose_color)
        elif index % 2:
            __draw_keypoint(overlay, keypoint, radius, __left_color)
        else:
            __draw_keypoint(overlay, keypoint, radius, __right_color)

    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


def draw_connections(
    frame: np.ndarray,
    keypoints: np.ndarray,
    thickness: int = 1,
    alpha: float = 1.0
) -> np.ndarray:
    __body_connections = [(5, 6), (5, 11), (6, 12), (11, 12)]
    __head_connections = [(0, 1), (0, 2), (1, 3), (2, 4)]
    __legs_connections = [(11, 13), (13, 15), (12, 14), (14, 16)]
    __hands_connections = [(5, 7), (7, 9), (6, 8), (8, 10)]

    overlay = frame.copy()
    for i, j in __body_connections:
        overlay = __draw_connection(
            overlay, keypoints[i], keypoints[j], thickness=thickness
        )
    for i, j in __head_connections:
        overlay = __draw_connection(
            overlay, keypoints[i], keypoints[j], thickness=thickness
        )
    for i, j in __legs_connections:
        overlay = __draw_connection(
            overlay, keypoints[i], keypoints[j], thickness=thickness
        )
    for i, j in __hands_connections:
        overlay = __draw_connection(
            overlay, keypoints[i], keypoints[j], thickness=thickness
        )

    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


def __draw_connection(
    frame: np.ndarray,
    keypoint1: np.ndarray,
    keypoint2: np.ndarray,
    color: tuple = (255, 255, 255),
    thickness: int = 1
) -> np.ndarray:
    x1, y1, v1 = keypoint1
    x2, y2, v2 = keypoint2
    if v1 > 0.4 and v2 > 0.4:
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, thickness, cv2.LINE_AA)
    return frame


def __draw_keypoint(
        frame: np.ndarray,
        keypoint: np.ndarray,
        radius: int,
        color: tuple
) -> np.ndarray:
    x, y, v = keypoint
    white = (255, 255, 255)
    if v > 0.4:
        cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), radius, white, 1, cv2.LINE_AA)


def draw_keypoints(
        rgb_image: np.ndarray,
        detection_result: mp.tasks.vision.PoseLandmarker
) -> np.ndarray:
    keypoints = PoseDetector.convert_landmarks_to_keypoints(detection_result)
    return draw_keypoints(rgb_image, keypoints)

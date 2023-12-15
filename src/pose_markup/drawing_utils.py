import cv2
import numpy as np
import mediapipe as mp


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from pose_markup.converting_utils import convert_landmarks_to_keypoints


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
        __draw_connection(
            overlay, keypoints[i], keypoints[j], thickness=thickness
        )
    for i, j in __head_connections:
        __draw_connection(
            overlay, keypoints[i], keypoints[j], thickness=thickness
        )
    for i, j in __legs_connections:
        __draw_connection(
            overlay, keypoints[i], keypoints[j], thickness=thickness
        )
    for i, j in __hands_connections:
        __draw_connection(
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
    x1, y1, _ = keypoint1
    x2, y2, _ = keypoint2
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
             color, thickness, cv2.LINE_AA)
    return frame


def __draw_keypoint(
        frame: np.ndarray,
        keypoint: np.ndarray,
        radius: int,
        color: tuple
) -> np.ndarray:
    x, y, _ = keypoint
    white = (255, 255, 255)

    cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (int(x), int(y)), radius, white, 1, cv2.LINE_AA)

    return frame


def draw_keypoints(
        rgb_image: np.ndarray,
        detection_result: mp.tasks.vision.PoseLandmarker
) -> np.ndarray:
    keypoints = convert_landmarks_to_keypoints(detection_result)
    return draw_keypoints(rgb_image, keypoints)

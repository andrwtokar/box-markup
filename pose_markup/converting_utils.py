import numpy as np
import mediapipe as mp


def convert_landmarks_to_keypoints(
    pose_result: mp.tasks.vision.PoseLandmarkerResult
) -> np.ndarray:
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

    landmarks = pose_result.pose_landmarks[0]
    mask_landmarks = [landmarks[i] for i in __mask_17_points.values()]
    return np.array([[landmark.x, landmark.y, landmark.visibility]
                     for landmark in mask_landmarks])


def unnormilized_keypoints(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    return (keypoints * [height, width, 1])

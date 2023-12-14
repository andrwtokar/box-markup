import os
import cv2
import ffmpeg
import numpy as np
import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


# Check and read Paths

INPUT_PATH = os.path.join(os.path.curdir, "data/")
OUTPUT_PATH = os.path.join(os.path.curdir, "output_data/")
MODEL_PATH = os.path.join(os.getcwd(), "model/pose_landmarker_full.task")

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)


# Create Landmarker options

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = BaseOptions(model_asset_path=MODEL_PATH)
options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.VIDEO
)


# Draw keypoints

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
    if v1 > 0.4 and v2 > 0.4:  # TODO: Change visaility settings
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
    if v > 0.4:  # TODO: Change visaility settings
        cv2.circle(frame, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), radius, white, 1, cv2.LINE_AA)


def draw_landmarks(rgb_image, detection_result):
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


# Landmark to keypoints

def landmark_to_keypoints(pose_result: mp.tasks.vision.PoseLandmarkerResult) -> np.ndarray:
    landmarks = pose_result.pose_landmarks[0]
    mask_17_points = {
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

    mask_landmarks = [landmarks[index] for index in mask_17_points.values()]
    return np.array([[landmark.x, landmark.y, landmark.visibility]
                     for landmark in mask_landmarks])


def unnormilized_keypoints(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    return (keypoints * [height, width, 1])


# Read and process files

names = os.listdir(INPUT_PATH)

with PoseLandmarker.create_from_options(options) as landmarker:
    name = names[0]

    # Create output directories
    output_dir = OUTPUT_PATH + name.split('.')[0] + "/"
    keypoints_dir = output_dir + "keypoints/"
    frames_dir = output_dir + "frames/"
    tmp_dir = output_dir + "tmp/"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(keypoints_dir):
        os.mkdir(keypoints_dir)

    if not os.path.exists(frames_dir):
        os.mkdir(frames_dir)

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    # read video
    videoclip = cv2.VideoCapture(INPUT_PATH + name)
    fps = videoclip.get(cv2.CAP_PROP_FPS)
    ms_per_frame = 1000 / fps
    frame_number = 0

    width = 0
    height = 0

    while videoclip.isOpened():
        flag, frame = videoclip.read()
        if not flag:
            break

        if frame_number == 0:
            width, height, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker_result = landmarker.detect_for_video(
            mp_image,
            int(ms_per_frame * frame_number)
        )
        keypoints = landmark_to_keypoints(landmarker_result)

        cv2.imwrite(
            frames_dir + f"{frame_number}.jpg",
            frame
        )
        cv2.imwrite(
            tmp_dir + f"{frame_number}.jpg",
            draw_pose(frame, unnormilized_keypoints(keypoints, width, height))
        )
        np.save(keypoints_dir + f"{frame_number}.npy", keypoints)

        # if cv2.waitKey(1) & 0xff == 27:  # exit if pressed `ESC`
        #     break

        # break  # Need to debug reading keypoints

        frame_number += 1

    videoclip.release()

    # os.system(
    #     (
    #         "ffmpeg " +
    #         "-i {}%01d.jpg " +
    #         "-r {} " +
    #         "-qscale:v 2 " +
    #         "-vcodec mpeg4 " +
    #         "-y {}result.mp4"
    #     ).format(tmp_dir, fps, output_dir)
    # )
    video = ffmpeg.input(tmp_dir + "%01d.jpg", framerate=fps)
    audio = ffmpeg.input(INPUT_PATH + name).audio

    ffmpeg.output(
        video,
        audio,
        output_dir + 'result.mp4',

        vcodec='mpeg4',
        **{'qscale:v': 2}
    ).run()

cv2.destroyAllWindows()

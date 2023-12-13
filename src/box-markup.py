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

def draw_keypoints(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    # copy it by MediaPipe example or code it by hands
    pass


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
    # TODO: Add filtering keypoints
    return [[landmark.x, landmark.y, landmark.visibility]
            for landmark in pose_result.pose_landmarks[0]]

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
    heigth = 0

    while videoclip.isOpened():
        flag, frame = videoclip.read()
        if not flag:
            break

        if frame_number == 0:
            width, heigth, _ = frame.shape

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
            draw_landmarks(frame, landmarker_result)
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

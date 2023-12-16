import os
import cv2
import numpy as np

from pose_markup.detector import PoseDetector
from pose_markup.video_utils import OutputFolders, create_result_video
from pose_markup.converting_utils import convert_landmarks_to_keypoints


class VideoProcessor:
    def __init__(self, detector: PoseDetector, output_dir: str) -> None:
        self.detector = detector
        self.output_dir = output_dir

    def predict_keypoints(self, input_name: str) -> OutputFolders:
        output_folders = OutputFolders(os.path.split(
            input_name)[1].split('.')[0], self.output_dir)
        videoclip = cv2.VideoCapture(input_name)
        fps = videoclip.get(cv2.CAP_PROP_FPS)
        ms_per_frame = 1000 / fps
        frame_number = 0

        while videoclip.isOpened():
            flag, frame = videoclip.read()
            if not flag:
                break

            landmarker_result = self.detector.predict_landmarks(
                frame, int(ms_per_frame * frame_number))
            keypoints = convert_landmarks_to_keypoints(landmarker_result)

            frame_filename = output_folders.frames_dir + \
                f"frame_{frame_number}.jpg"
            cv2.imwrite(frame_filename, frame)

            # Uncomment next lines if you wanna draw landmarks by mediapipe.solutions utily
            # tmp_frame_filename = output_folders.tmp_dir + \
            #     f"frame_{frame_number}.jpg"
            # tmp_frame = draw_landmarks(frame, landmarker_result)
            # cv2.imwrite(tmp_frame_filename, tmp_frame)

            keypoints_filename = output_folders.keypoints_dir + \
                f"frame_{frame_number}.npy"
            np.save(keypoints_filename, keypoints)

            frame_number += 1

        videoclip.release()

        return output_folders

    def process_video(self, input_name: str):
        output_folders = self.predict_keypoints(input_name)
        output_video = create_result_video(input_name, output_folders)

        output_video.run()

    def __create_landmarker_options(model_path: str):
        pass

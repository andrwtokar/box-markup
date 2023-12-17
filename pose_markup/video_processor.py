import os
import cv2
import numpy as np
import mediapipe as mp

from pose_markup.detector import PoseDetector
from pose_markup.video_utils import OutputFolders, add_keypoints_to_frames, create_result_video
from pose_markup.converting_utils import convert_landmarks_to_keypoints


class VideoProcessor:
    def __init__(self, detector_model_path: str, output_dir: str) -> None:
        self.detector_model_path = detector_model_path
        self.output_dir = output_dir

    def predict_keypoints(self, input_name: str) -> OutputFolders:
        detector = PoseDetector(self.detector_model_path)
        output_folders = OutputFolders(os.path.split(
            input_name)[1].split('.')[0], self.output_dir)
        videoclip = cv2.VideoCapture(input_name)

        frame_count = int(videoclip.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number_size = len(str(frame_count))

        fps = videoclip.get(cv2.CAP_PROP_FPS)
        ms_per_frame = 1000 / fps
        frame_number = 0

        while videoclip.isOpened():
            flag, frame = videoclip.read()
            if not flag:
                break

            landmarker_result = detector.predict_landmarks(
                frame, int(ms_per_frame * frame_number))
            keypoints = convert_landmarks_to_keypoints(landmarker_result)

            frame_filename = output_folders.frames_dir + \
                f"frame_{frame_number:0{frame_number_size}}.jpg"
            cv2.imwrite(frame_filename, frame)

            # Uncomment next lines if you wanna draw landmarks by mediapipe.solutions utily
            # tmp_frame_filename = output_folders.tmp_dir + \
            #     f"frame_{frame_number}.jpg"
            # tmp_frame = draw_landmarks(frame, landmarker_result)
            # cv2.imwrite(tmp_frame_filename, tmp_frame)

            keypoints_filename = output_folders.keypoints_dir + \
                f"frame_{frame_number:0{frame_number_size}}.npy"
            np.save(keypoints_filename, keypoints)

            frame_number += 1

        videoclip.release()

        return output_folders

    def process_video(self, input_name: str):
        output_folders = self.predict_keypoints(input_name)
        output_folders = add_keypoints_to_frames(output_folders)
        output_video = create_result_video(input_name, output_folders)

        output_video.run()

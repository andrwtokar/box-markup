import os
import cv2
import ffmpeg
import numpy as np

from pose_markup.detector import PoseDetector
from pose_markup.drawing_utils import draw_landmarks, draw_pose
from pose_markup.converting_utils import convert_landmarks_to_keypoints


class OutputFolders:
    def __init__(self, name: str, output_dir: str) -> None:
        self.output_dir = output_dir + name + "/"

        self.keypoints_dir = self.output_dir + "keypoints/"
        self.frames_dir = self.output_dir + "frames/"
        self.tmp_dir = self.output_dir + "tmp/"

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        if not os.path.exists(self.keypoints_dir):
            os.mkdir(self.keypoints_dir)

        if not os.path.exists(self.frames_dir):
            os.mkdir(self.frames_dir)

        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)


class VideoProcessor:
    def __init__(
        self,
        detector: PoseDetector,
        output_dir: str
    ) -> None:
        self.detector = detector
        self.output_dir = output_dir

    def predict_keypoints(self, input_name: str) -> OutputFolders:
        output_folders = OutputFolders(
            os.path.split(input_name)[1].split('.')[0],
            self.output_dir
        )
        videoclip = cv2.VideoCapture(input_name)
        fps = videoclip.get(cv2.CAP_PROP_FPS)
        ms_per_frame = 1000 / fps
        frame_number = 0

        while videoclip.isOpened():
            flag, frame = videoclip.read()
            if not flag:
                break

            landmarker_result = self.detector.predict_landmarks(
                frame,
                int(ms_per_frame * frame_number)
            )
            keypoints = convert_landmarks_to_keypoints(
                landmarker_result
            )

            cv2.imwrite(
                output_folders.frames_dir + f"frame_{frame_number}.jpg",
                frame
            )
            # Uncomment it if you wanna draw landmarks by mediapipe.solutions utily
            # cv2.imwrite(
            #     output_folders.tmp_dir + f"frame_{frame_number}.jpg",
            #     draw_landmarks(frame, landmarker_result)
            # )
            np.save(
                output_folders.keypoints_dir + f"frame_{frame_number}.npy",
                keypoints
            )

            frame_number += 1

        videoclip.release()

        return output_folders

    def add_keypoints_to_frames(self, output_folders: OutputFolders):
        frame_names = os.listdir(output_folders.frames_dir)
        keypoints_names = os.listdir(output_folders.keypoints_dir)

        for frame_name, keypoints_name in zip(frame_names, keypoints_names):
            frame = cv2.imread(frame_name)
            keypoints = np.load(keypoints_name)
            result_frame = draw_pose(frame, keypoints)

            cv2.imwrite(output_folders.tmp_dir + frame_name, result_frame)

        return output_folders

    def create_result_video(
            self,
            input_name: str,
            output_folders: OutputFolders
    ) -> ffmpeg.Stream:
        videoclip = cv2.VideoCapture(input_name)
        fps = videoclip.get(cv2.CAP_PROP_FPS)
        videoclip.release()

        video = ffmpeg.input(
            output_folders.tmp_dir + "frame_%01d.jpg",
            framerate=fps
        )
        audio = ffmpeg.input(input_name).audio

        return ffmpeg.output(
            video,
            audio,
            output_folders.output_dir + 'result.mp4',
            vcodec='mpeg4',
            **{'qscale:v': 2}
        )

    def process_video(self, input_name: str):
        output_folders = self.predict_keypoints(input_name)
        output_video = self.create_result_video(input_name, output_folders)

        output_video.run()

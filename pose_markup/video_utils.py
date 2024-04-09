import cv2
import os
import ffmpeg
import numpy as np

from pose_markup.drawing_utils import draw_pose


class OutputFolders:
    def __init__(self, name: str, output_dir: str) -> None:
        self.output_dir = output_dir + name + "/"

        self.keypoints_dir = os.path.join(self.output_dir, "keypoints/")
        self.frames_dir = os.path.join(self.output_dir, "frames/")
        self.tmp_dir = os.path.join(self.output_dir, "tmp/")

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        if not os.path.exists(self.keypoints_dir):
            os.mkdir(self.keypoints_dir)

        if not os.path.exists(self.frames_dir):
            os.mkdir(self.frames_dir)

        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    @staticmethod
    def __remove_directory(path: str):
        for root, dirs, files in os.walk(path):
            # For each file in the directory
            for file in files:
                # Construct the full path to the file
                file_path = os.path.join(root, file)
                # Delete the file
                os.remove(file_path)
            # For each subdirectory in the directory
            for dir in dirs:
                # Construct the full path to the subdirectory
                dir_path = os.path.join(root, dir)
                # Delete the subdirectory
                OutputFolders.__remove_directory(dir_path)
        # Delete the top-level directory
        os.rmdir(path)


def add_keypoints_to_frames(output_folders: OutputFolders):
    frame_names = sorted(os.listdir(output_folders.frames_dir))
    keypoints_names = sorted(os.listdir(output_folders.keypoints_dir))

    for frame_name, keypoints_name in zip(frame_names, keypoints_names):
        frame = cv2.imread(output_folders.frames_dir + frame_name)
        keypoints = np.load(output_folders.keypoints_dir + keypoints_name)
        result_frame = draw_pose(frame, keypoints)

        cv2.imwrite(output_folders.tmp_dir + frame_name, result_frame)

    return output_folders


def create_result_video(input_name: str, output_folders: OutputFolders) -> ffmpeg.Stream:
    add_keypoints_to_frames(output_folders)
    videoclip = cv2.VideoCapture(input_name)
    fps = videoclip.get(cv2.CAP_PROP_FPS)
    videoclip.release()

    video = ffmpeg.input(
        output_folders.tmp_dir + "frame_*.jpg",
        pattern_type="glob",
        framerate=fps
    )
    audio = ffmpeg.input(input_name).audio

    output_name = output_folders.output_dir + 'result.mp4'

    if os.path.exists(output_name):
        os.remove(output_name)

    return ffmpeg.output(video, audio, output_name,
                         vcodec='mpeg4', **{'qscale:v': 2})

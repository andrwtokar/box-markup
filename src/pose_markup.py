import os
from detector import PoseDetector
from video_processing import VideoProcessor

INPUT_PATH = os.path.join(os.path.curdir, "data/")
OUTPUT_PATH = os.path.join(os.path.curdir, "output_data/")
MODEL_PATH = os.path.join(os.getcwd(), "model/pose_landmarker_full.task")

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

detector = PoseDetector(MODEL_PATH)
video_proc = VideoProcessor(detector, OUTPUT_PATH)

names = os.listdir(INPUT_PATH)

video_proc.process_video(INPUT_PATH + names[0])

import sys
sys.path.append('./')

import os
from pose_markup.detector import PoseDetector
from pose_markup.video_processor import VideoProcessor

INPUT_PATH = os.path.join(os.getcwd(), "data/")
OUTPUT_PATH = os.path.join(os.getcwd(), "output_data/")
MODEL_PATH = os.path.join(os.getcwd(), "model/pose_landmarker_full.task")

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

detector = PoseDetector(MODEL_PATH)
video_proc = VideoProcessor(detector, OUTPUT_PATH)

names = os.listdir(INPUT_PATH)

print(f"\n\n{names[::1]}")

for name in names:
    video_proc.process_video(INPUT_PATH + name)

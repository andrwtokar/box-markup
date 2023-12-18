import os
import sys
if "examples" in os.getcwd():
    sys.path.append('../')
else:
    sys.path.append('./')

from pose_markup.video_processor import VideoProcessor
from pose_markup.detector import PoseDetector


INPUT_PATH = os.path.join(os.getcwd(), "data/")
OUTPUT_PATH = os.path.join(os.getcwd(), "output_data/")
MODEL_PATH = os.path.join(os.getcwd(), "model/pose_landmarker_full.task")

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

video_proc = VideoProcessor(MODEL_PATH, OUTPUT_PATH)

names = os.listdir(INPUT_PATH)

for name in names:
    video_proc.process_video(INPUT_PATH + name)

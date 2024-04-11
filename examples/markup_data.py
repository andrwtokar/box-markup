import os
import sys
if "examples" in os.getcwd():
    sys.path.append('../')
else:
    sys.path.append('./')

import cv2
import numpy as np

from pose_markup.markup_utils import MarkupImage

# TODO: Добавить вывод правил редактирования
# TODO: Добавить ввод названия видео с клавиатуры

OUTPUT_DIR = "output_data/1_2/frames/"
KEYPOINTS_DIR = "output_data/1_2/keypoints/"


window_name = "test_frame"
frame_names = sorted(os.listdir(OUTPUT_DIR))
keypoints_names = sorted(os.listdir(KEYPOINTS_DIR))

number_of_frames = len(frame_names)
for ind, (frame_filename, keypoints_filename) in enumerate(zip(frame_names, keypoints_names)):
    frame_filepath = frames_dir + frame_filename
    keypoints_filepath = keypoints_dir + keypoints_filename

    frame = cv2.imread(frame_filepath)
    keypoints = np.load(keypoints_filepath)
    markup_image = MarkupImage(frame, keypoints)

    if markup_image.run(window_name, f"{ind + 1}/{number_of_frames}"):
        break

    # Comment out this line if you don't need to save updates keypoints
    np.save(keypoints_filepath, markup_image.get_keypoints())

cv2.destroyAllWindows()

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
cv2.namedWindow(window_name)
for frame_name, keypoints_name in zip(frame_names, keypoints_names):
    frame = cv2.imread(OUTPUT_DIR + frame_name)
    keypoints = np.load(KEYPOINTS_DIR + keypoints_name)
    markup_image = MarkupImage(frame, keypoints)

    if markup_image.run(window_name):
        break

    # TODO: Записывать обновленные координаты в файл

cv2.destroyAllWindows()

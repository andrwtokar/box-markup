import os
import sys
if "examples" in os.getcwd():
    sys.path.append('../')
else:
    sys.path.append('./')

import cv2
import numpy as np

from pose_markup.markup_utils import MarkupImage

print("""### MARKING KEYPOINT ON VIDEO.
### Instructions for frame-by-frame marking of the data on video:
###  - Directory with data have to contain folders with frames ('frames/') and and with keypoints ('keypoints/');
###  - Click LBM on the window and move it on the screen. If you click on the keypoint
###    you will move keypoint to point where you release the LBM;
###  - Push Space or Enter to go to the next frame;
###  - Push Esc to end program.""")
output_dir = input("Enter path to directory with data: ")

frames_dir = output_dir + "/frames/"
keypoints_dir = output_dir + "/keypoints/"


window_name = "test_frame"
frame_names = sorted(os.listdir(frames_dir))
keypoints_names = sorted(os.listdir(keypoints_dir))

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

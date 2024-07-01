import os
import sys
if "examples" in os.getcwd():
    sys.path.append('../')
else:
    sys.path.append('./')

import cv2
import numpy as np

from pose_markup.drawing_utils import draw_pose

OUTPUT_DIR = "output_data/1_2/frames/"
KEYPOINTS_DIR = "output_data/1_2/keypoints/"

KEYPOINT_INDEX_TO_NAME = {
    0: "nose",
    1: "left eye",
    2: "right eye",
    3: "left ear",
    4: "right ear",

    5: "left shoulder",
    6: "right shoulder",
    7: "left elbow",
    8: "right elbow",
    9: "left wrist",
    10: "right wrist",

    11: "left hip",
    12: "right hip",
    13: "left knee",
    14: "right knee",
    15: "left ankle",
    16: "right ankle"
}

frame_name = "frame_0004"

frame = cv2.imread(OUTPUT_DIR + frame_name + ".jpg")
keypoints = np.load(KEYPOINTS_DIR + frame_name + ".npy")
keypoints[:, :2] = keypoints[:, :2].astype(np.int32).astype(np.float64)


def search_near_keypoint(x, y):
    min_dist = np.inf
    keypoint_ind = -1

    for ind, keypoint in enumerate(keypoints):
        dist = np.sqrt((keypoint[0] - x) ** 2 + (keypoint[1] - y) ** 2)
        if dist < min_dist and dist < 4:
            min_dist = dist
            keypoint_ind = ind

    if keypoint_ind == -1:
        return None

    return keypoint_ind


last_points = []


def print_point_coordinates(event, x, y, flags, param):
    near_keypoint = search_near_keypoint(x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left button down: {x=} {y=}")
        if near_keypoint is None:
            print("Near Keypoint: None")
        else:
            print(f"Near Keypoint: {KEYPOINT_INDEX_TO_NAME[near_keypoint]}")
        last_points.append(near_keypoint)
        print(f"num of nearest points = {len(last_points)}")
        print()
    elif event == cv2.EVENT_LBUTTONUP:
        last_point = last_points.pop()
        print(f"Left button up: {x=} {y=}")
        if last_point is None:
            print("last buttom keypoint = None")
        else:
            print(
                f"last buttom keypoint = {KEYPOINT_INDEX_TO_NAME[last_point]}")
        print(f"num of nearest points = {len(last_points)}")
        if last_point is not None:
            keypoints[last_point][0] = x
            keypoints[last_point][1] = y
        print()


cv2.namedWindow("test_frame")
cv2.setMouseCallback("test_frame", print_point_coordinates)
while True:
    printed_frame = draw_pose(frame, keypoints)
    cv2.imshow("test_frame", printed_frame)
    print("Before waitKey")
    key = cv2.waitKey(0) & 0xFF
    print(f"{key=} and char={chr(key)}")
    print("After waitKey")
    print()

    if key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()

import cv2
import numpy as np
from pose_markup.drawing_utils import draw_pose


def search_near_keypoint(keypoints, x, y):
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


class MarkupImage:
    def __init__(self, image, keypoints) -> None:
        self.image = image
        self.keypoints = keypoints
        self.last_keypoint = []

    def run(self, window_name):
        def on_click(event, x, y, flags, param):
            near_keypoint = search_near_keypoint(self.keypoints, x, y)

            if event == cv2.EVENT_LBUTTONDOWN:
                self.last_keypoint.append(near_keypoint)
            if event == cv2.EVENT_LBUTTONUP:
                last_point = self.last_keypoint.pop()
                if last_point is not None:
                    self.keypoints[last_point][0] = x
                    self.keypoints[last_point][1] = y

            cv2.imshow(window_name, draw_pose(self.image, self.keypoints))

        # TODO: Выводить номер изображения в видео для ориентации
        # TODO: Выбрать как показывать кадры: все в одном окне или в разных.
        cv2.setMouseCallback(window_name, on_click)
        while True:
            cv2.imshow(window_name, draw_pose(self.image, self.keypoints))
            key = cv2.waitKey()

            if key == 13 or key == 32:
                break
            elif key == 27:
                return True

        return False

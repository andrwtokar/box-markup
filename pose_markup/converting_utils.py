import numpy as np
import mediapipe as mp

from skimage import measure
from shapely.geometry import Polygon, MultiPolygon


__mask_landmarks_to_coco = {
    0: 0,  # nose
    1: 2,  # left eye
    2: 5,  # right eye
    3: 7,  # left ear
    4: 8,  # right ear

    5: 11,  # left shoulder
    6: 12,  # right shoulder
    7: 13,  # left elbow
    8: 14,  # right elbow
    9: 15,  # left wrist
    10: 16,  # right wrist

    11: 23,  # left hip
    12: 24,  # right hip
    13: 25,  # left knee
    14: 26,  # right knee
    15: 27,  # left ankle
    16: 28  # right ankle
}


def filter_keypoints_to_coco(keypoints: np.ndarray) -> np.ndarray:
    return np.array([keypoints[i] for i in __mask_landmarks_to_coco.values()])


def convert_landmarks_to_keypoints(pose_result: mp.tasks.vision.PoseLandmarkerResult) -> np.ndarray:
    return np.array([[landmark.x, landmark.y, landmark.visibility]
                     for landmark in pose_result.pose_landmarks[0]])


def convert_visability_to_coco(keypoints: np.ndarray, visability_threshold: float) -> np.ndarray:
    # Set visability in COCO format:
    #  - v=0: not labeled (in which case x=y=0)
    #  - v=1: labeled but not visible (visability less then threshold)
    #  - v=2: labeled and visible (visability more or equal then threshold)
    for keypoint in keypoints:
        if (keypoint[2] > visability_threshold):
            keypoint[2] = 2
        else:
            keypoint[2] = 1

    return keypoints


def polygons_from_mask(sub_mask: np.ndarray) -> list[Polygon]:
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(
        sub_mask, 0.5, positive_orientation='low')

    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0)
        polygons.append(poly)

    return polygons


def segmantations_from_polygons(polygons: list[Polygon]) -> list:
    return [np.array(poly.exterior.coords).ravel().round(2).tolist()
            for poly in polygons]


def create_coco_annotation(keypoints: np.ndarray, seg_mask: np.ndarray) -> dict:
    polygons = polygons_from_mask(seg_mask)
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds

    return {
        'segmentations': segmantations_from_polygons(polygons),
        'keypoints': keypoints.flatten().astype(int).tolist(),
        'num_keypoints': len(keypoints),
        'iscrowd': 0,
        'bbox': [round(x, 2), round(y, 2), round(max_x - x, 2), round(max_y - y, 2)],
        'area': round(multi_poly.area, 4)
    }

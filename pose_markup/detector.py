import time
import mediapipe as mp
import numpy as np

from pose_markup.converting_utils import (
    convert_landmarks_to_keypoints, convert_visability_to_coco, create_coco_annotation, filter_keypoints_to_coco)


class PoseDetector:
    def __init__(self, model_path: str) -> None:
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        base_options = BaseOptions(model_asset_path=model_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.VIDEO,
            output_segmentation_masks=True)

        self.landmarker = PoseLandmarker.create_from_options(options)
        self.total_prediction_time_ms = 0.0
        self.total_num_of_frames = 0

        self.visability_threshold = 0.6
        self.prev_frame_keypoints = []
        self.prev_seg_mask = []

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def predict_landmarks(self, frame: np.ndarray, timestamp_ms: int) -> mp.tasks.vision.PoseLandmarkerResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def predict_coco_keypoints(self, frame: np.ndarray, timestamp_ms: int) -> dict:
        start = time.time()
        height, width = frame.shape[:2]
        landmark_result = self.predict_landmarks(frame, timestamp_ms)

        keypoints = []
        if len(landmark_result.pose_landmarks) == 0:
            # Take keypoints from the previous frame to fill the current one
            keypoints = self.prev_frame_keypoints
        else:
            keypoints = convert_landmarks_to_keypoints(landmark_result)
            keypoints = filter_keypoints_to_coco(keypoints)
            keypoints = convert_visability_to_coco(
                keypoints, self.visability_threshold)

            # Save the current keypoints for use in the next frame
            self.prev_frame_keypoints = keypoints
            keypoints = keypoints * [width, height, 1]

        seg_mask = []
        if landmark_result.segmentation_masks is None:
            # Take segmantation mask from the previous frame to fill the current one
            seg_mask = self.prev_seg_mask
        else:
            np_mask = landmark_result.segmentation_masks[0].numpy_view() * 255
            seg_mask = np.zeros(np_mask.shape).astype(float)
            seg_mask[np_mask > 127] = 255.0

            # Save the current segmentation mask for use in the next frame
            self.prev_seg_mask = seg_mask

        ann = create_coco_annotation(keypoints, seg_mask)
        end = time.time()

        self.total_prediction_time_ms += (end - start) * 1000
        self.total_num_of_frames += 1

        return ann

    def get_average_prediction_time(self):
        return round(self.total_prediction_time_ms / self.total_num_of_frames, 3)

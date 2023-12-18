# pose-markup

---

Project for marking poses on video.

To recognize points, a Pose Landmarker from MediaPipe is used.
The output data has 17 key points located in accordance with the COCO topology.

## Prepequirements 

- Firstly you need to install ffmpeg to your device. You can do it by [link](https://ffmpeg.org/download.html).
- Secondly install requirements by next command:
  ```
   pip install -r requirements.txt
  ```

## Run markup data

### Download model

Download [landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models) to directory `model/`.

### Add raw data

Add the videos to be processed to the directory `date/`. Further, this folder will use in processing.

### Process data

```
python3 examples/keypoints_markup.py
```

### Take processed data

When the program finishes executing, you can find the result of the work in directory `output_data/`.

# box-markup

---

Project for markup video with training boxing punches. Here you can lay out key points according to the COCO topography using the OpenPose model.

## Run 

### Install requirements

- Firstly you need to install ffmpeg to your device. You can do it by [link](https://ffmpeg.org/download.html).
- Secondly install requirements by next command:
    ```
    pip install -r requirements.txt
    ```

### Add raw data

Add the videos to be processed to the directory `date/`. Further, this folder will use in processing.

### Run processing

```
python3 src/box-markup.py
```

### Take processed data

When the program finishes executing, you can find the result of the work in directory `output_data/`.

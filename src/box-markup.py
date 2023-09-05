import subprocess
from openpose.utils import draw_body_connections, draw_keypoints
from openpose.body.estimator import BodyPoseEstimator
import json
import ffmpeg
import cv2
import os

from tqdm import tqdm

INPUT_PATH = os.path.join(os.path.curdir, "data/")
OUTPUT_PATH = os.path.join(os.path.curdir, "output_data/")


def ffprobe_stdout_json(file_path) -> str:
    command_array = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path
    ]

    result = subprocess.run(command_array, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True)

    return result.stdout


def run_predict_pipeline(input_name: str, output_name: str) -> None:
    videoclip = cv2.VideoCapture(input_name)
    info = json.loads(ffprobe_stdout_json(input_name))
    videoinfo = [i for i in info["streams"] if i["codec_type"] == "video"][0]
    stream = None

    while videoclip.isOpened():
        flag, frame = videoclip.read()
        if not flag:
            break

        keypoints = estimator(frame)
        frame = draw_body_connections(frame, keypoints, thickness=2, alpha=0.7)
        frame = draw_keypoints(frame, keypoints, radius=4, alpha=0.8)

        if stream is None:
            input_framesize = frame.shape[:2]
            stream = (
                ffmpeg
                .input(
                    'pipe:',
                    format='rawvideo',
                    pix_fmt="bgr24",
                    s='%sx%s' % (input_framesize[1], input_framesize[0]),
                    r=videoinfo["avg_frame_rate"]
                )
                .output(
                    output_name,
                    pix_fmt=videoinfo["pix_fmt"],
                    vcodec=videoinfo["codec_name"]
                )
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

        stream.stdin.write(frame)

        if cv2.waitKey(20) & 0xff == 27:  # exit if pressed `ESC`
            break

    videoclip.release()
    stream.stdin.close()
    stream.wait()
    cv2.destroyAllWindows()


if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

names = os.listdir(INPUT_PATH)
estimator = BodyPoseEstimator(pretrained=True)

for name in tqdm(names):
    input_name = INPUT_PATH + name
    output_name = OUTPUT_PATH + name

    if os.path.exists(output_name):
        os.remove(output_name)

    run_predict_pipeline(input_name, output_name)

#from curses import window
import cv2
import os
import numpy as np
from datetime import datetime
import glob
import sys
from pathlib import Path
#from (root directory) import (py file)
from tqdm import tqdm
#from utils import load_pth_file

#指定した秒数のフレームを画像として保存
def save_frames_range_sec(
    video_path, 
    start_sec, stop_sec, step_sec, 
    dir_path, 
    basename, ext="jpg"
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_inv = 1 / fps

    sec = start_sec

    n_steps = (stop_sec - start_sec) // step_sec
    tmp_ = [None] * n_steps

    pbar = tqdm(tmp_, total=n_steps)
    for _ in pbar:
        n = round(fps * sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if ret:
            cap_img_resize = cv2.resize(
                            frame, 
                            dsize=(1920, 1080), 
                            fx=0, fy=0, 
                            interpolation=cv2.INTER_AREA
                            )
            cv2.imwrite(
                "{}_{}_{:.2f}.{}".format(
                    base_path, 
                    str(n).zfill(digit), 
                    n * fps_inv, 
                    ext
                ),
                cap_img_resize,
            )
        sec += ( step_sec * 30 ) 

def main():
    # root = Path(r"E:/senkouka/drone_video")
    dir_path = "D:/senkouka/drone_imgs"
    basename = "drone_img"

    save_frames_range_sec(
        "D:/senkouka/drone_video/drone_flight_0967",
        38,
        96,
        1,
        dir_path,
        basename,
    )

    save_frames_range_sec(
        "D:/senkouka/drone_video/drone_flight_0967",
        102,
        103,
        1,
        dir_path,
        basename,
    )

    save_frames_range_sec(
        "D:/senkouka/drone_video/DJI_0048",
        0,
        232,
        1,
        dir_path,
        basename,
    )

    save_frames_range_sec(
        "D:/senkouka/drone_video/DJI_0062",
        0,
        125,
        1,
        dir_path,
        basename,
    )

#　【必須】メインの関数を実行するために
if __name__ == "__main__":
    main()

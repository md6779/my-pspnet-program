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
                            dsize=(1920, 1080) , 
                            fx=0, fy=0, 
                            interpolation=cv2.INTER_AREA
                            )
            cv2.imwrite(
                "{}_{}_{:.2f}.{}".format(
                    base_path, str(n).zfill(digit), 
                    n * fps_inv, ext
                ),
                cap_img_resize,
            )
        sec += ( step_sec * 30 ) 

def save_all_frames(
    video_path, 
    dir_path, 
    basename, 
    ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cap_img_resize = cv2.resize(
                frame, 
                dsize=(1920, 1080) , 
                fx=0, fy=0, 
                interpolation=cv2.INTER_AREA
                )
            cv2.imwrite(
                '{}_{}.{}'.format(
                    base_path, 
                    str(n).zfill(digit), 
                    ext), 
                cap_img_resize)
            n += 1
        else:
            return

#リサイズされたことを確認する
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

    # while sec < stop_sec:
    #     print(sec)
    #     n = round(fps * sec)
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    #     ret, frame = cap.read()
    #     if ret:
    #         cv2.imwrite(
    #             "{}_{}_{:.2f}.{}".format(
    #                 base_path, str(n).zfill(digit), 
    #                 n * fps_inv, ext
    #             ),
    #             frame,
    #         )
    #     else:
    #         return
    #     sec += step_sec 

# #切り出した画像を動画化する
# def jpg_to_mp4(dir_path_1,  name1):

#     jpg_imgs = []

#     jpg_target_imgs = sorted(glob.glob(f"{dir_path_1}/*.jpg"))
#     jpg_target_imgs = jpg_target_imgs[:]

#     pbar = tqdm(jpg_target_imgs)
#     for filename in pbar:
#         img = cv2.imread(filename)
#         height, width, layers = img.shape
#         global size
#         size = (width, height)
#         jpg_imgs.append(img)

#     out_1 = cv2.VideoWriter(name1, 
#                         cv2.VideoWriter_fourcc(*"mp4v"),
#                         5.0, size)

#     for jpg_img in jpg_imgs:
#         out_1.write(jpg_img)
#     out_1.release()

# def png_to_mp4(dir_path_2, name2):

#     png_imgs = []

#     png_target_imgs = sorted(glob.glob(f"{dir_path_2}/*.png"))
#     png_target_imgs = png_target_imgs[:]

#     pbar = tqdm(png_target_imgs)
#     for filename in pbar:
#         img = cv2.imread(filename)
#         height, width, layers = img.shape
#         global size
#         size = (width, height)
#         png_imgs.append(img)

#     out_2 = cv2.VideoWriter(name2, 
#                         cv2.VideoWriter_fourcc(*"mp4v"),
#                         5.0, size)

#     for  png_img in  png_imgs:
#         out_2.write(png_img)
#     out_2.release()

# #画像化された結果を表示する
# def show_mp4 (dir_path_1, 
#             ext = 'mp4', delay = 1, 
#             window_name_1 = 'frame1',
#             ):
#     video_1 = cv2.VideoCapture(f"{dir_path_1}")
#     if not video_1.isOpened():
#         sys.exit()    
    
#     while True:
#         ret1, frame1 = video_1.read()
#         ret = ret1 
        
#         if ret == True:
#             frame_resize_1 = cv2.resize(frame1, 
#                                         dsize = (1920, 1080))
#             cv2.imshow(window_name_1, 
#                         frame_resize_1)
#             cv2.waitKey(0)
#             if cv2.waitKey(delay) & 0xFF == ord('q'):
#                 break
#             else:
#                 video_1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
#     video_1.release()
#     cv2.destroyWindow(window_name_1)

#            dir_path_2, 
#            window_name_2 = 'frame2'
#    video_2 = cv2.VideoCapture(f"{dir_path_2}")
#        ret2, frame2 = video_2.read()
#            frame_resize_2= cv2.resize(frame2, 
#                                        dsize = (1920, 1080))
        # cv2.imshow(window_name, frame)
#            cv2.imshow(window_name_2, 
#                        frame_resize_2)
#                video_2.set(cv2.CAP_PROP_POS_FRAMES, 0)
# #    video_2.release()
#    cv2.destroyWindow(window_name_2)

def main():
    # root = Path(r"E:/senkouka/drone_video")
    dir_path = Path(r"E:/senkouka/drone_imgs")
    basename = "sample_video_img"

    # if not dir_path.exists():
        # dir_path.mkdir(parents=True)

    # for file_path in (root.glob(".mp4")):
        # if file_path.is_dir():
            # continue
        # print(file_path)
        # save_all_frames(file_path, dir_path, basename)

    save_frames_range_sec(
        "E:/senkouka/drone_video",
        38,
        96,
        1,
        dir_path,
        basename,
    )

    save_frames_range_sec(
        "E:/senkouka/drone_video",
        102,
        103,
        1,
        dir_path,
        basename,
    )

    save_frames_range_sec(
        "E:/senkouka/drone_video",
        102,
        103,
        1,
        dir_path,
        basename,
    )

    save_frames_range_sec(
        "E:/senkouka/drone_video",
        102,
        103,
        1,
        dir_path,
        basename,
    )
    
#　【必須】メインの関数を実行するために
if __name__ == "__main__":
    main()

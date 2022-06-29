#from curses import window
import cv2
import os
import numpy as np
from datetime import datetime
import glob
import sys
#from (root directory) import (py file)
from tqdm import tqdm
#from utils import load_pth_file

def jpg_to_mp4(dir_path_1,  name1):

    jpg_imgs = []

    jpg_target_imgs = sorted(glob.glob(f"{dir_path_1}/*.jpg"))
    jpg_target_imgs = jpg_target_imgs[:]

    pbar = tqdm(jpg_target_imgs)
    for filename in pbar:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        global size
        size = (width, height)
        jpg_imgs.append(img)

    out_1 = cv2.VideoWriter(name1, 
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        5.0, size)

    for jpg_img in jpg_imgs:
        out_1.write(jpg_img)
    out_1.release()

def png_to_mp4(dir_path_2, name2):

    png_imgs = []

    png_target_imgs = sorted(glob.glob(f"{dir_path_2}/*.png"))
    png_target_imgs = png_target_imgs[:]

    pbar = tqdm(png_target_imgs)
    for filename in pbar:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        global size
        size = (width, height)
        png_imgs.append(img)

    out_2 = cv2.VideoWriter(name2, 
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        5.0, size)

    for  png_img in  png_imgs:
        out_2.write(png_img)
    out_2.release()

def show_mp4 (dir_path_1, 
            ext = 'mp4', delay = 1, 
            window_name_1 = 'frame1',
            ):
    video_1 = cv2.VideoCapture(f"{dir_path_1}")
    if not video_1.isOpened():
        sys.exit()    
    
    while True:
        ret1, frame1 = video_1.read()
        ret = ret1 
        
        if ret == True:
            frame_resize_1 = cv2.resize(frame1, 
                                        dsize = (1920, 1080))
            cv2.imshow(window_name_1, 
                        frame_resize_1)
            cv2.waitKey(0)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            else:
                video_1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    video_1.release()
    cv2.destroyWindow(window_name_1)

def main():
    filename = "E:/senkouka/result_4/jpg_to_video.mp4"
    dir_path = "E:/senkouka/result_4"
    jpg_to_mp4(dir_path, filename)
    show_mp4(filename)

if __name__ == "__main__":
    main()
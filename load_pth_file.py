from importlib.resources import path
import os
import glob
import torch
import torch.cuda
import numpy as np 
import cv2
import utils
import os
from datetime import datetime
import sys
#from (root directory) import (py file)
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision import transforms
from tqdm import tqdm
from config import Config
from model.pspnet import PSPNet
from utils import dataset, trans
from pathlib import Path
from utils import mp4_to_jpg
from utils import jpg_to_mp4
# from torchvideotransforms import video_transforms, volume_transforms
from threading import Thread

def get_paths(input_dir, exts=None):
    paths = sorted([x for x in input_dir.iterdir()])
    if exts:
        paths = list(filter(lambda x: x.suffix in exts, 
                paths))

    return paths

# def video_data(path):
#     img = np.array(Image.open(path).convert("RGB"), 
#                     dtype=np.float32)
#     img_tmp = np.array(Image.new(mode="P", 
#                         size=(473, 473)),
#                         dtype=np.float32)

#     # transform = transforms.Compose([transforms.ToTensor()])
#     transform = trans.Compose([trans.ToTensor()])
    
#     img, _= transform(img, img_tmp)
#     return img

#画像入力
def img_data(path):
    img = np.array(Image.open(path).convert("RGB"), 
                    dtype=np.float32)
    img_tmp = np.array(Image.new(mode="P", 
                        size=(473, 473)),
                        dtype=np.float32)

    # transform = transforms.Compose([transforms.ToTensor()])
    transform = trans.Compose([trans.ToTensor()])
    
    img, _= transform(img, img_tmp)
    return img

def pspnet_load(ckpt_path):
    ckpt_path = r"E:\senkouka\semseg-pspnet\exp\2022-06-14_09h52m13s\ckpt\model_30.pth"
    classes_name =  ["background", "land", "debris"]
    name = "output"
    model = PSPNet(
        layers=50,
        classes=len(classes_name),
        zoom_factor=8,
        pretrained=False,
    )
    print(f"Loading checkpoint '{ckpt_path}' ...")
    model.load_state_dict(torch.load(ckpt_path))
    device = torch.device("cuda" if torch.cuda.is_available() 
                                else "cpu")
    model.to(device)

def img_pspnet_eval(img_path):
# パラメータ?
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = [item * value_scale for item in mean]
    std = [item * value_scale for item in std]
    pspnet_load.model.eval()
# 画像入力    
    for path in get_paths(img_path, 
                exts=[".jpg", ".jpeg", ".png"]):
        path = Path(path)
        img = img_data(path)
        x = F.resize(img, size=[473, 473])
        x = F.normalize(x, mean=mean, std=std)
        x = x.to(pspnet_load.device).float()
        with torch.no_grad():
            output = pspnet_load.model(x.unsqueeze(0))
        output = output.max(1)[1]

def main():
    img_path = Path("E:/senkouka/result_4")
    output_path = Path("E:/senkouka/output_1")
    ckpt_path = r"E:\senkouka\semseg-pspnet\exp\2022-06-14_09h52m13s\ckpt\model_30.pth"
    
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    classes_name =  ["background", "land", "debris"]
    name = "output"
#　PSPNetの重みを呼び出す
    pspnet_load(ckpt_path)
    img_pspnet_eval(img_path)
# パレット
    palette =  [
    0, 0, 0, # rgb(0, 0, 0)
    128, 0, 128, # rgb(128, 0, 128)
    220, 128, 0, # rgb(220, 128, 0)
    ]
#　オリジナル画像とα画像の比較処理
    img = F.to_pil_image(img_pspnet_eval.output.to(torch.uint8))
    img.putpalette(palette)    
    orig_img = Image.open(path)
    img = img.resize(orig_img.size)
    alpha_img = Image.blend(orig_img.convert("RGBA"), 
                img.convert("RGBA"), alpha=0.5)
    
    name = path.stem
    orig_color_dst_fp = output_path / f"{name}_orig.jpg"

    alpha_dst_fp = output_path / f"{name}_alpha.png"
    alpha_img.copy().resize(
                    (1920, 1080)
                    ).save(
                    alpha_dst_fp
                    )
    orig_img.copy().resize(
                    (1920, 1080)
                    ).convert("RGB").save(
                    orig_color_dst_fp
                    )
    
    alpha_filename = "E:/senkouka/output_1/alphapng.mp4"
    jpg_to_mp4.png_to_mp4(output_path, alpha_filename)
    jpg_to_mp4.show_mp4(alpha_filename)
    

if __name__ == "__main__":
    global dir_path, alpha_filename

    alpha_path = "E:/senkouka/output_1"
    alpha_filename = "E:/senkouka/output_1/alphapng.mp4"

    mp4_to_jpg.main()
    main()

#    vid_path = "E:/senkouka/drone_flight_0967.MP4"
# img_path = "E:/senkouka/result_4"
# #    basename = "sample_video_img"
#    video_filename = "E:/senkouka/result_4/jpg_to_video.mp4"
# mp4_to_jpg.png_to_mp4(alpha_path, alpha_filename)
# mp4_to_jpg.show_mp4(alpha_filename)

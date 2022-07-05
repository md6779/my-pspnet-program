from importlib.resources import path
import os
import glob
from typing import get_type_hints
from regex import P
import torch
import torch.cuda
import numpy as np 
import cv2
import utils
import os
from datetime import datetime
import sys
#from (root directory) import (py file)
import PIL
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torchvision import datasets as D
from torchvision.utils import save_image
import albumentations as A
from tqdm import tqdm
#from config import Config
from model.pspnet import PSPNet
from utils import trans
from pathlib import Path
import random 

def read_imgs():

    imgs = cv2.imread("")
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGBA)

def read_masks():
    mask_1 = cv2.imread("")
    mask_2 = cv2.imread("")
    mask_3 = cv2.imread("")
    masks = [mask_1, mask_2, mask_3]

def img_transform():

    size = (width, height) = (473, 473)   
    if random.random() > 0.5:
        P = 0.5
        return P         
    var_limit = (10, 100)
    mean = 0
    per_channel = True
    always_apply = False
    blur_limit = (3, 9)
    sigma_limit = 0
    fog_coef = (lower, upper) = (0.3, 1.0)
    alpha_coef = 0.1

    transform = A.Compose(
        A.RandomCrop(size),
        A.HorizontalFlip(P),
        A.RandomBrightnessContrast(P),
        A.GaussianBlur(blur_limit, sigma_limit, P),
        A.GaussNoise(
            var_limit, mean, 
            per_channel, always_apply, P
            ),
        A.RandomFog(fog_coef, alpha_coef, always_apply, P)
    )

    transformed_1 = transform(imgs=read_imgs.imgs, 
    mask= read_masks.masks)[""]
    transformed_2 = transform(imgs=read_imgs.imgs, 
    mask= read_masks.masks)[""]
    transformed_3 = transform(imgs=read_imgs.imgs, 
    mask= read_masks.masks)[""]


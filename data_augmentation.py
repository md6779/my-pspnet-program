from importlib.resources import path
import os
import glob
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
from torchvision import transforms as T, datasets as D
from tqdm import tqdm
#from config import Config
from model.pspnet import PSPNet
from utils import trans
from pathlib import Path
import random 

class rotation :

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(
            self.angles
            )
        return F.rotate(x, angle)

class brightness:
    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, x):
        brightness_factor = random.choice(
            self.brightness_factor
            )
        return F.adjust_brightness(x, brightness_factor)



def dataset_get_imgs(dataset_root_dir=f''):
    dataset = D.ImageFolder(dataset_root_dir)

def rand_img_data_transforms(dataset_root_dir=f''):
    rotation_transform = rotation(
        angles=[-30, -15, 0, 15, 30]
    )
    brightness_transform = brightness(
        brightness_factor=[1,2,3]
    )
    datatransform  = T.Compose([
        rotation_transform, 
        brightness_transform
        ])
    dataset_augmented = D.ImageFolder(
        dataset_root_dir, 
        transform=datatransform
        )

def img_data_transforms(dataset_root_dir=f''):
    size = (473, 473)
    degrees = (0, 180)
    brightness = (0.25, 2.0)    
    contrast = (0.25, 2.0)
    saturation = (0.25, 2.0)
    sharpness_factor = (0, 2)
    hue = None
    P = 0.5
    data_transform = T.Compose([
        T.FiveCrop(size = size),
        T.TenCrop(size = size, vertical_flip=True),
        T.RandomCrop(size = size),
        T.ColorJitter(brightness, 
        contrast = contrast, 
        saturation = saturation, 
        hue = hue),
        T.RandomAdjustSharpness(sharpness_factor, P),
        T.RandomRotation(degrees),
        T.RandomHorizontalFlip(P), 
        T.RandomVerticalFlip(P)])
    applier = T.RandomApply(
        torch.nn.ModuleList[
        data_transform
    ],P)
    imgs_transforms = [applier(dataset_get_imgs) 
        for _ in range()]
    dataset_augmented = D.ImageFolder(dataset_root_dir)



if __name__ == "__main__":
    dataset_root_dir=f''
    dataset_get_imgs(dataset_root_dir)
    img_data_transforms(dataset_root_dir)

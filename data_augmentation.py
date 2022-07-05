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
from torchvision import transforms as T, datasets as D
from torchvision.utils import save_image
import albumentations as A
from tqdm import tqdm
#from config import Config
from model.pspnet import PSPNet
from utils import trans
from pathlib import Path
import random 

class rotation :

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, imgs):
        angle = random.choice(
            self.angles
            )
        return F.rotate(imgs, angle)

    def angles():
        if random.random() > 0.5 :
            angles = random.randint(-45, 45) 
            return angles

class brightness:

    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, imgs):
        brightness_factor = random.choice(
            self.brightness_factor
            )
        return F.adjust_brightness(imgs, brightness_factor)

    def brightnessfactor():
        if random.random() > 0.5 :
            brightness_factor = random.randint(1, 5)
            return brightness_factor

class contrast:

    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, imgs, contrast_factor) :
        contrast_factor = random.choice(
                self.contrast_factor
                )
        return F.adjust_contrast(imgs, contrast_factor)

    def contrastfactor():
        if random.random() > 0.5 :
            contrast_factor = random.randint(1, 5)
            return contrast_factor   

class saturation:

    def __init__(self, saturation_factor):
        self.saturation_factor = saturation_factor

    def __call__(self, imgs, saturation_factor) :
        saturation_factor = random.choice(
                self.saturation_factor
                )
        return F.adjust_saturation(imgs, saturation_factor)

    def saturationfactor():
        if random.random() > 0.5 :
            saturation_factor = random.randint(1, 5)
            return saturation_factor   

class ten_crop:

    def __init__(self, size, vertical_flip) :
        self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, imgs, size, vertical_flip):
        size = random.choice(
            self.size
        )
        vertical_flip = random.choice(
            self.vertical_flip
        )
        return F.ten_crop(imgs, size, vertical_flip)

def dataset_get_imgs(dataset_root_dir=f''):
    dataset = D.ImageFolder(dataset_root_dir)

def rand_img_data_transforms(imgs, 
    dataset_root_dir=f'', 
    ):
    size = (473, 473)
    rotation_transform = rotation(
        imgs, 
        rotation.angles)
    brightness_transform = brightness(
        imgs, 
        brightness.brightnessfactor)
    contrast_transform = contrast(
        imgs, 
        contrast.contrastfactor)
    ten_crop_transform = ten_crop(
        imgs, 
        size, 
        vertical_flip = True)
    saturation_transform = saturation(
        imgs,
        saturation.saturation_factor
    )
    datatransform = T.Compose(
        [rotation_transform, 
        brightness_transform, 
        ten_crop_transform, 
        contrast_transform]
    )
    dataset_augmented = D.ImageFolder(
        dataset_root_dir, 
        transform=datatransform
        )

# def img_data_transforms(dataset_root_dir=f''):
#     size = (473, 473)
#     degrees = (0, 180)
#     brightness = (0.25, 2.0)    
#     contrast = (0.25, 2.0)
#     saturation = (0.25, 2.0)
#     sharpness_factor = (0, 2)
#     hue = None
#     P = 0.5
#     data_transform = T.Compose([
#         T.TenCrop(size = size, vertical_flip=True),
#         T.ColorJitter(brightness = brightness, 
#         contrast = contrast, 
#         saturation = saturation, 
#         hue = hue),
#         ])
#     if random.random() > P:
#         rand_data_transform = T.Compose([
#             T.RandomCrop(size = size),
#             T.RandomAdjustSharpness(sharpness_factor, P),
#             T.RandomRotation(degrees),
#             T.RandomHorizontalFlip(P)])
#     applier = T.RandomApply(
#         torch.nn.ModuleList[
#         data_transform,
#         rand_data_transform
#     ],P)
#     imgs_transforms = [applier(dataset_get_imgs) 
#         for _ in range()]
#     dataset_augmented = D.ImageFolder(dataset_root_dir)

    # save_image("<TENSOR>", "EXPORT_FILEPATH")


if __name__ == "__main__":
    dataset_root_dir=f''
    dataset_get_imgs(dataset_root_dir)
    rand_img_data_transforms(dataset_root_dir)

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

def dataset_get_imgs(dataset_root_dir=f''):
    dataset = D.ImageFolder(dataset_root_dir, )

def img_data_transforms(dataset_root_dir=f''):
    size = (473, 473)
    degrees = (0, 180)
    brightness = (0.25, 2.0)    
    contrast = (0.25, 2.0)
    saturation = (0.25, 2.0)
    hue = None
    data_transform = T.Compose([
        T.FiveCrop(size = size),
        T.RandomCrop(size = size),
        T.ColorJitter(brightness, 
        contrast = contrast, 
        saturation = saturation, 
        hue = hue),
        T.RandomRotation(degrees)])
    applier = T.RandomApply(
        torch.nn.ModuleList[
        data_transform
    ],p=0.5)
    imgs_transforms = [applier(dataset_get_imgs) 
        for _ in range()]
    dataset_augmented = D.ImageFolder(dataset_root_dir )



if __name__ == "__main__":
    dataset_root_dir=f''
    dataset_get_imgs(dataset_root_dir)

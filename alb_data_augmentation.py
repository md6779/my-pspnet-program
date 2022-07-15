from typing import get_type_hints
from regex import P
import numpy as np 
import cv2
from torchvision.utils import save_image
import albumentations as A
from tqdm import tqdm
#from config import Config
from model.pspnet import PSPNet
from utils import trans
from pathlib import Path
import random 
from collections import OrderedDict as OD

#関数化したいところ
#変換処理をするところ
#保存先のところ
#イメージとマスク

def augmented():
    # augmented_bright = brightness_contrast(
    #                     image=imgs, mask=seg
    #                 )
    # augmented_adv_blur = adv_blur(image=imgs, mask=seg)
    # augmented_blur = gauss_blur(image=imgs, mask=seg)
    # augmented_noise = gauss_noise(image=imgs, mask=seg)
    # augmented_sharp = sharp(image=imgs, mask=seg)
    # augmented_flip = flip(image=imgs, mask=seg)
    # augmented_gray = grayscale(image=imgs, mask=seg)

    # grayscale_img = transform_img(grayscale, imgs, seg)
        augmented_image_bright = augmented_bright['image']
        augmented_image_blur = augmented_blur ['image']
        augmented_image_adv_blur = augmented_adv_blur['image']
        augmented_image_noise = augmented_noise['image']
        augmented_image_sharp = augmented_sharp['image']
        augmented_image_flip = augmented_flip['image']
        augmented_image_gray = augmented_gray['image']
        augmented_mask_bright = augmented_bright['mask']
        augmented_mask_blur = augmented_blur['mask']
        augmented_mask_adv_blur = augmented_adv_blur['mask']
        augmented_mask_noise = augmented_noise['mask']
        augmented_mask_sharp = augmented_sharp['mask']
        augmented_mask_flip = augmented_flip['mask']
        augmented_mask_gray = augmented_gray['mask']

        augmented_image_list = [
            augmented_image_bright,
            augmented_image_blur,
            augmented_image_adv_blur,
            augmented_image_noise,
            augmented_image_sharp,
            augmented_image_flip,
            augmented_image_gray
        ]
        augmented_mask_list = [
            augmented_mask_bright,
            augmented_mask_blur, 
            augmented_mask_adv_blur,
            augmented_mask_noise,
            augmented_mask_sharp,
            augmented_mask_flip,
            augmented_mask_gray
        ]


def add_noise(img):

    # Getting the dimensions of the image
    row , col = img.shape
    
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
    
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
        
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
        
        # Color that pixel to white
        img[y_coord][x_coord] = 255
        
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
    
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
        
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
        
        # Color that pixel to black
        img[y_coord][x_coord] = 0
        
    return img

def noise_img():
    
    # salt-and-pepper noise can
    # be applied only to grayscale images
    # Reading the color image in grayscale image
    img = cv2.imread('img', cv2.IMREAD_GRAYSCALE)

    #Storing the image
    cv2.imwrite('snp_img', add_noise(img))

def noisy(noise_typ,image):

    if noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

def transform():
    transforms_dict: OD[str, list] = OD(
        flip = A.HorizontalFlip(always_apply=False, p=1.0),
        brightness_contrast = A.RandomBrightnessContrast(
                                brightness_limit=0.25,
                                contrast_limit=0.25,
                                brightness_by_max=True,
                                always_apply=False, p=0.5
                                ),
        #ごみをよりはっきり見える
        sharp = A.Sharpen(
                alpha=(0.2, 0.5), 
                lightness=(0.45, 1.0), 
                always_apply=False, p=0.5),
        #ごみをはっきり見えないとき
        adv_blur = A.AdvancedBlur(
                    blur_limit=(3, 7), 
                    sigmaX_limit=(0.2, 1.0), 
                    sigmaY_limit=(0.2, 1.0), 
                    rotate_limit=90, 
                    beta_limit=(0.5, 8.0), 
                    noise_limit=(0.9, 1.1), 
                    always_apply=False, p=0.5
                    ),
        gauss_blur = A.GaussianBlur(
                    blur_limit=(3, 9),
                    sigma_limit=0,
                    always_apply=False, p=0.5
                    ),
        #ノイズ付加
        gauss_noise = A.GaussNoise(
                    var_limit=(10, 100),    
                    mean=0,
                    per_channel=True,
                    always_apply=False, p=0.5
                    ),
        grayscale = A.ToGray(p = 1.0),
        invert = A.InvertImg(p = 1.0),
        snp_noise = noisy("s&p", noise_img),
        poisson = noisy("poisson", noise_img)
    )
    
    #BPF(船，雲，波，ゴミ)
    #白だけ残すフィルタ
    # SnP
    #Invert

def transform_img(transform, img, mask):
    return transform(image=img, mask=mask)

def main():
    root = Path(r"D:\senkouka\test_imgs\images")
    output_path = Path(r"D:\senkouka\test_augmented_8")

    if not output_path.exists():
        output_path.mkdir(parents=True)

    #10種類の変改を目指す
    #表に書く必要がある
    # 左右反転
    flip = A.HorizontalFlip(always_apply=False, p=1.0)
    brightness_contrast = A.RandomBrightnessContrast(
                            brightness_limit=0.25,
                            contrast_limit=0.25,
                            brightness_by_max=True,
                            always_apply=False, p=0.5
                            )
    #ごみをよりはっきり見える
    sharp = A.Sharpen(
            alpha=(0.2, 0.5), 
            lightness=(0.45, 1.0), 
            always_apply=False, p=0.5)
    #ごみをはっきり見えないとき
    adv_blur = A.AdvancedBlur(
                blur_limit=(3, 7), 
                sigmaX_limit=(0.2, 1.0), 
                sigmaY_limit=(0.2, 1.0), 
                rotate_limit=90, 
                beta_limit=(0.5, 8.0), 
                noise_limit=(0.9, 1.1), 
                always_apply=False, p=0.5
                ) 
    gauss_blur = A.GaussianBlur(
                blur_limit=(3, 9),
                sigma_limit=0,
                always_apply=False, p=0.5
                )
    #ノイズ付加
    gauss_noise = A.GaussNoise(
                var_limit=(10, 100),    
                mean=0,
                per_channel=True,
                always_apply=False, p=0.5
                ) 
    grayscale = A.ToGray(p = 1.0)

    #BPF(船，雲，波，ゴミ)
    #白だけ残すフィルタ
    # SnP
    #Invert

    for file_path in (root.glob("*.jpg")):   
        if file_path.is_dir():
            continue
        print(file_path)
        img = cv2.imread(str(file_path))
        img_red = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_color_list = [img, img_red]

        # print(file_path.parent)
        mask_path = file_path.parent.parent / f"segmentations/{file_path.stem}.png"
        seg = cv2.imread(str(mask_path))
        print(mask_path)

        for imgs in img_color_list:
            augmented_bright = brightness_contrast(
                                image=imgs, mask=seg
                            )
            augmented_adv_blur = adv_blur(image=imgs, mask=seg)
            augmented_blur = gauss_blur(image=imgs, mask=seg)
            augmented_noise = gauss_noise(image=imgs, mask=seg)
            augmented_sharp = sharp(image=imgs, mask=seg)
            augmented_flip = flip(image=imgs, mask=seg)
            augmented_gray = grayscale(image=imgs, mask=seg)


        augmented_image_bright = augmented_bright['image']
        augmented_image_blur = augmented_blur ['image']
        augmented_image_adv_blur = augmented_adv_blur['image']
        augmented_image_noise = augmented_noise['image']
        augmented_image_sharp = augmented_sharp['image']
        augmented_image_flip = augmented_flip['image']
        augmented_image_gray = augmented_gray['image']
        augmented_mask_bright = augmented_bright['mask']
        augmented_mask_blur = augmented_blur['mask']
        augmented_mask_adv_blur = augmented_adv_blur['mask']
        augmented_mask_noise = augmented_noise['mask']
        augmented_mask_sharp = augmented_sharp['mask']
        augmented_mask_flip = augmented_flip['mask']
        augmented_mask_gray = augmented_gray['mask']

        augmented_image_list = [
            augmented_image_bright,
            augmented_image_blur,
            augmented_image_adv_blur,
            augmented_image_noise,
            augmented_image_sharp,
            augmented_image_flip,
            augmented_image_gray
        ]
        augmented_mask_list = [
            augmented_mask_bright,
            augmented_mask_blur, 
            augmented_mask_adv_blur,
            augmented_mask_noise,
            augmented_mask_sharp,
            augmented_mask_flip,
            augmented_mask_gray
        ]

        for i, (AugImg , AugMask) in enumerate(
            zip(
                augmented_image_list, 
                augmented_mask_list
                ), 1):
            cv2.imwrite(
                str(output_path / f"{file_path.stem}.jpg"),
                img
            )
            cv2.imwrite(
                str(output_path / f"{file_path.stem}_augmented_{i}.jpg"), 
                AugImg
            )
            cv2.imwrite(
                str(output_path / f"{file_path.stem}.png"),
                seg
            )
            cv2.imwrite(
                str(output_path / f"{file_path.stem}_augmented_mask_{i}.png"), 
                AugMask
            )

if __name__ == "__main__":
    main()


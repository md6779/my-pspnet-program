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
    fog_coef = (lower, upper) = (0.3, 0.9)
    alpha_coef = 0.1

    imgs_masks_transform = A.Compose(
        A.RandomCrop(size),
        A.HorizontalFlip(P),      
    )
    
    imgs_transform = A.Compose(
        A.RandomBrightnessContrast(P),
        A.GaussianBlur(blur_limit, sigma_limit, P),
        A.GaussNoise(
            var_limit, mean, 
            per_channel, always_apply, P
            ),
        A.RandomFog(fog_coef, alpha_coef, always_apply, P)
    )

    # for i in range (3):
    #     transform[i] = (
    #         imgs_masks_transform(
    #         imgs=read_imgs.imgs, 
    #         mask= read_masks.masks)[""],
    #         imgs_transform(
    #         imgs=read_imgs.imgs, 
    #         mask= read_masks.masks)[""]
    #         )

def main():
    root = Path(r"E:\senkouka\data_dataset_voc\images")
    output_path = Path(r"E:\senkouka\augmented_2")

    if not output_path.exists():
        output_path.mkdir(parents=True)

    # imgs_transform_1 = A.Compose()
    imgs_flip = A.Compose(
        [A.HorizontalFlip(always_apply=False, p=0.5)]
    )
    imgs_bright = A.Compose(
        [A.RandomBrightnessContrast(
                        brightness_limit=0.23,
                        contrast_limit=0.23,
                        brightness_by_max=True,
                        always_apply=False, p=0.5)
        ]
    )
    imgs_transform = A.OneOf(
            [A.Sequential([
                    A.GaussianBlur(blur_limit=(3, 9),sigma_limit=0,p=0.5),
                    A.GaussNoise(
                        var_limit=(10, 100),    
                        mean=0,
                        per_channel=True,
                        always_apply=False, p=0.5
                )       
            ])
        ]
    )
    # imgs_fog = A.RandomFog(
    #         fog_coef_lower=0.3, 
    #         fog_coef_upper=0.9, 
    #         alpha_coef=0.1, 
    #         always_apply=False, p=0.5
    # )

    # print(root)
    for file_path in (root.glob("*.jpg")):   
        if file_path.is_dir():
            continue
        print(file_path)
        img = cv2.imread(str(file_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

        # print(file_path.parent)
        mask_path = file_path.parent.parent / f"segmentations/{file_path.stem}.png"
        seg = cv2.imread(str(mask_path))
        print(mask_path)

        augmented = imgs_transform(image=img, mask=seg)
        augmented_flip = imgs_flip(image=img, mask=seg)
        augmented_bright = imgs_bright(image=img, mask=seg)
            # imgs_noise(image=img, mask=seg),
            # imgs_fog(image=img, mask=seg)

        augmented_image = augmented['image']
        augmented_image_flip = augmented_flip['image']
        augmented_image_bright = augmented_bright['image']
        augmented_mask = augmented['mask']
        augmented_mask_flip = augmented_flip['mask']
        augmented_mask_bright = augmented_bright['mask']

        augmented_args = [
            augmented_image,
            augmented_image_flip,
            augmented_image_bright,
            augmented_mask,
            augmented_mask_flip,
            augmented_mask_bright]
        
        cv2.imwrite(
            str(output_path / f"{file_path.stem}_augmented.jpg"), 
            augmented_image
            )
        cv2.imwrite(
            str(output_path / f"{file_path.stem}_augmented_flip.jpg"), 
            augmented_image_flip
            ) 
        cv2.imwrite(
            str(output_path / f"{file_path.stem}_augmented_bright.jpg"), 
            augmented_image_bright
            )    
        cv2.imwrite(
            str(output_path / f"{file_path.stem}_augmented.png"), 
            augmented_mask
            )
        cv2.imwrite(
            str(output_path / f"{file_path.stem}_augmented_flip.png"), 
            augmented_mask_flip
            )
        cv2.imwrite(
            str(output_path / f"{file_path.stem}_augmented_bright.png"), 
            augmented_mask_bright
            )
        # seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGBA)



    # imgs = cv2.imread("")
    # imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGBA)
    # img_transform()


if __name__ == "__main__":
    main()


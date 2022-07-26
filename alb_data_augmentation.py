import typing
from typing import Any, Iterable
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

def transform_img(transform, img, mask):
    return transform(image=img, mask=mask)

def get_transformed_properties(
    transformed_data: dict[str, Any]
) -> typing.tuple[Any, typing.list[Iterable[float]], typing.list[int]]:
    return transformed_data["image"], transformed_data["mask"]

def get_transform_options(transform_data: typing.dict[str, Any]) -> dict:
    return {
        "image": transform_data["image"],
        "mask": transform_data["mask"],
    }

def get_compose(transforms_dict: typing.dict[str, typing.list]) -> Iterable[typing.tuple[str, A.Compose]]:
    for key, transform in transforms_dict.items():
        yield key, A.Compose(transform)

def save_process(transformed_data: typing.dict[str, Any], save_root: Path, stem: str) -> None:
    tfd_img, tfd_mask = get_transformed_properties(transformed_data)
    # img = transformed_data["imaga"]
    # mask = transformed_data["mask"]


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

def noise_img():
    root = Path(r"E:\senkouka\test_imgs\images")
    output_path = Path(r"E:\senkouka\test_augmented_8")
    
    if not output_path.exists():
        output_path.mkdir(parents=True)

    # salt-and-pepper noise can
    # be applied only to grayscale images
    # Reading the color image in grayscale image
    img = cv2.imread('img', cv2.IMREAD_GRAYSCALE)

    #Storing the image
    cv2.imwrite(f'{output_path}_snp_img', noisy(img))

def main():
    root = Path(r"E:\senkouka\test_imgs\images")
    output_path = Path(r"E:\senkouka\test_augmented_8")

    if not output_path.exists():
        output_path.mkdir(parents=True)

    transforms_dict: OD[str, typing.list] = OD(
        flip = A.HorizontalFlip(always_apply=False, p=1.0),
        brightness_contrast = A.RandomBrightnessContrast(
                                brightness_limit=0.25,
                                contrast_limit=0.25,
                                brightness_by_max=True,
                                always_apply=False, p=1.0
                                ),
        #ごみをよりはっきり見える
        sharp = A.Sharpen(
                alpha=(0.2, 0.5), 
                lightness=(0.45, 1.0), 
                always_apply=False, p=1.0),
        #ごみをはっきり見えないとき
        adv_blur = A.AdvancedBlur(
                    blur_limit=(3, 7), 
                    sigmaX_limit=(0.2, 1.0), 
                    sigmaY_limit=(0.2, 1.0), 
                    rotate_limit=90, 
                    beta_limit=(0.5, 8.0), 
                    noise_limit=(0.9, 1.1), 
                    always_apply=False, p=1.0
                    ),
        gauss_blur = A.GaussianBlur(
                    blur_limit=(3, 9),
                    sigma_limit=0,
                    always_apply=False, p=1.0
                    ),
        #ノイズ付加
        gauss_noise = A.GaussNoise(
                    var_limit=(10, 100),    
                    mean=0,
                    per_channel=True,
                    always_apply=False, p=1.0
                    ),
        grayscale = A.ToGray(p = 1.0),
        invert = A.InvertImg(p = 1.0),
    # SnP
        snp_noise = noisy("s&p", noise_img),
        poisson = noisy("poisson", noise_img)
    )
    
    #BPF(船，雲，波，ゴミ)
    #白だけ残すフィルタ
    #Invert

    for file_path in (root.glob("*.jpg")):   
        if file_path.is_dir():
            continue
        print(file_path)
        img = cv2.imread(str(file_path))

        # print(file_path.parent)
        mask_path = file_path.parent.parent / f"segmentations/{file_path.stem}.png"
        seg = cv2.imread(str(mask_path))
        print(mask_path)

        base_transform_data = {
            "image": img,
            "mask": seg
        }
        for key, transforms in get_compose(transforms_dict):
            base_transformed = transforms(get_transformed_properties(base_transform_data))

            save_process(
                        base_transformed,   
                        save_root=output_path, 
                        stem=f"{file_path.stem}_augmented_{key}"
                        )

if __name__ == "__main__":
    main()


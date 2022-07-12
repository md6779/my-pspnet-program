from __future__ import annotations
from dataclasses import dataclass

import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import albumentations as A
import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass
class BBoxFormatConversion:
    """Convert format between 'albumentations' and 'yolo'."""

    height: int
    width: int
    target_format: str = "yolo"  # yolo, coco, pascal_voc, albumentations

    def convert_target_to_albu(self, bboxes: Iterable) -> Iterable:
        return A.bbox_utils.convert_bboxes_to_albumentations(
            bboxes, source_format=self.target_format, rows=self.width, cols=self.height
        )

    def convert_albu_to_target(self, albu_bboxes: Iterable) -> Iterable:
        return A.bbox_utils.convert_bboxes_from_albumentations(
            albu_bboxes, target_format=self.target_format, rows=self.width, cols=self.height
        )

    def update_hw(self, height: int, width: int) -> None:
        self.height = height
        self.width = width


def parse_anno_txt(txt_fp: Path) -> tuple[list[list[float]], list[int]]:
    bboxes: list[list[float]] = []
    class_ids: list[int] = []

    bbox_list = txt_fp.read_text().strip().split("\n")
    for line in bbox_list:
        cls_id, *bbox = line.split(" ")
        bbox = list(map(float, bbox))
        bboxes.append(bbox)
        class_ids.append(int(cls_id))
    return bboxes, class_ids


def get_compose(transforms_dict: dict[str, list]) -> Iterable[tuple[str, A.Compose]]:
    for key, transform in transforms_dict.items():
        yield key, A.Compose(transform)


def get_transformed_properties(
    transformed_data: dict[str, Any]
) -> tuple[Any, list[Iterable[float]], list[int]]:
    return transformed_data["image"], transformed_data["bboxes"], transformed_data["class_ids"]


def save_process(transformed_data: dict[str, Any], save_root: Path, stem: str) -> None:
    tfd_img, tfd_bboxes, class_ids = get_transformed_properties(transformed_data)
    # img = transformed_data["imaga"]
    # mask = transformed_data["mask"]

    # image
    img_pil = Image.fromarray(tfd_img).convert("RGB")
    img_save_fp = save_root / f"{stem}.jpg"
    img_pil.save(img_save_fp)

    # annotation
    lines: list[str] = []
    for bbox, class_id in zip(tfd_bboxes, class_ids):
        bbox_str = " ".join(list(map(str, bbox)))
        lines.append(f"{class_id} {bbox_str}\n")
    txt_save_fp = save_root / f"{stem}.txt"
    txt_save_fp.write_text("".join(lines))


def get_transform_options(transform_data: dict[str, Any]) -> dict:
    return {
        "image": transform_data["image"],
        "bboxes": transform_data["bboxes"],
        "class_ids": transform_data["class_ids"],
    }


def main() -> None:
    # root = Path("data_aug_data")
    root = Path(r"C:\okamura\labelimg_workspace\images")
    # save_root = Path(r"D:\workspace\data_augmentation")
    save_root = Path(r"C:\okamura\labelimg_workspace\data_augmented")
    save_root.mkdir(parents=True, exist_ok=True)

    # fixed seed
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    # * settings for albumentations -----
    # fmt: off
    # enumerate single transforms without `A.Compose`
    base_transforms_dict: OrderedDict[str, list] = OrderedDict(
        gaussian_blur=[A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=1)],  # default, p=1
        glass_blur=[A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, p=1)],  # default, p=1
        median_blur=[A.MedianBlur(blur_limit=7, p=1)],  # default, p=1
        clahe=[A.CLAHE(clip_limit=4, tile_grid_size=(8, 8), p=1)],  # default, p=1
        sharp=[A.Sharpen(alpha=(0.2, 0.5), p=1)],  # default, p=1
        gamma=[A.RandomGamma(gamma_limit=(80, 120), eps=None, p=1)],  # default, p=1
        brightness_contrast=[A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=1)],  # default, p=1
        noise=[A.GaussNoise(var_limit=(10, 50), mean=0, per_channel=True, p=1)],  # default, p=1
        multiply_noise=[A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=False, p=1)],  # default, p=1
        grayscale=[A.ToGray(p=1)],  # default, p=1
        downscale=[A.Downscale(scale_min=0.25, p=1)],  # default, p=1
        posterize=[A.Posterize(num_bits=4, p=1)],  # default, p=1
    )

    rotate90_rain_transform = A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.RandomRain(
                slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(220, 220, 220), blur_value=3, brightness_coefficient=0.7, rain_type=None, p=0.5
            ),
        ]
    )
    hflip_transform = A.Compose([A.HorizontalFlip(p=1)])
    # fmt: on
    # * ---------------------------------
    bbox_formatter = BBoxFormatConversion(height=416, width=416)

    exts = ["jpg", "png", "webp", "gif"]
    img_list = [x for ext in exts for x in root.glob(f"*.{ext}")]
    # img_list = img_list[:1]

    pbar = tqdm(img_list, desc="Augment")
    for img_fp in pbar:
        img = np.array(Image.open(img_fp).convert("RGB"))  # for supporting japanese filename
        bboxes, class_ids = parse_anno_txt(img_fp.with_suffix(".txt"))
        h, w, _ = img.shape
        bbox_formatter.update_hw(h, w)

        # convert 'yolo fmt' to 'albu fmt'
        albu_bboxes = bbox_formatter.convert_target_to_albu(bboxes=bboxes)

        base_transform_data = {
            "image": img,
            "bboxes": albu_bboxes,
            "class_ids": class_ids,
        }

        for key, transform in get_compose(base_transforms_dict):
            # * base transforms
            base_transformed = transform(**get_transform_options(base_transform_data))

            # * rotation 90 degrees transform
            rotate_transformed = rotate90_rain_transform(**get_transform_options(base_transformed))
            # copy for saving
            rotate_transformed_cp = rotate_transformed.copy()
            # convert 'albu fmt' to 'yolo fmt' and update bboxes
            rotate_transformed_cp["bboxes"] = bbox_formatter.convert_albu_to_target(
                albu_bboxes=rotate_transformed_cp["bboxes"]
            )
            # save
            save_process(rotate_transformed_cp, save_root, stem=f"{img_fp.stem}_{key}")

            # * horizontal transform
            hflip_transformed = hflip_transform(**get_transform_options(rotate_transformed))
            # copy for saving
            hflip_transformed_cp = hflip_transformed.copy()
            # convert 'albu fmt' to 'yolo fmt' and update bboxes
            hflip_transformed_cp["bboxes"] = bbox_formatter.convert_albu_to_target(
                hflip_transformed_cp["bboxes"]
            )
            # save
            save_process(hflip_transformed_cp, save_root, stem=f"{img_fp.stem}_{key}_hflip")


if __name__ == "__main__":
    main()

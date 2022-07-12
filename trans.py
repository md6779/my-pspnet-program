from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch import Tensor


@dataclass
class Compose(object):
    transform: Sequence[Callable]

    def __call__(self, img, label) -> tuple[Tensor, Tensor]:
        for t in self.transform:
            img, label = t(img, label)
        return img, label


@dataclass
class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, img, label):
        if not isinstance(img, np.ndarray) or not isinstance(label, np.ndarray):
            raise RuntimeError(
                "segtransform.ToTensor() only handle np.ndarray"
                "[eg: data readed by cv2.imread()].\n"
            )
        if len(img.shape) > 3 or len(img.shape) < 2:
            raise RuntimeError(
                "segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"
            )
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        if not len(label.shape) == 2:
            raise RuntimeError(
                "segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"
            )

        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()

        return img, label


class Normalize(object):
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, img, label):
        if self.std is None:
            for t, m in zip(img, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(img, self.mean, self.std):
                t.sub_(m).div_(s)
        return img, label


@dataclass
class Resize(object):
    size: tuple[int, int]

    def __call__(self, img, label) -> tuple:
        img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, self.size[::-1], interpolation=cv2.INTER_NEAREST)
        return img, label


@dataclass
class RandomScale(object):
    scale: tuple[float, float]
    aspect_ratio: Optional[tuple[float, float]] = None

    def __call__(self, img, label) -> tuple:
        tmp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        tmp_aspect_ratio = 1.0
        if self.aspect_ratio:
            tmp_aspect_ratio = math.sqrt(
                self.aspect_ratio[0]
                + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()  # noqa
            )
        scale_factor_x = tmp_scale * tmp_aspect_ratio
        scale_factor_y = tmp_scale / tmp_aspect_ratio
        img = cv2.resize(
            img, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR
        )
        label = cv2.resize(
            label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST
        )
        return img, label


@dataclass
class Crop(object):
    size: Union[int, tuple[int, int]]
    crop_type: str = "center"
    padding: Optional[list[float]] = None
    ignore_label: int = 255

    crop_h: int = field(init=False)
    crop_w: int = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.size, int):
            self.crop_h = self.size
            self.crop_w = self.size
        else:
            self.crop_h, self.crop_w = self.size

        if self.padding and len(self.padding) != 3:
            raise RuntimeError("padding channel is not equal with 3\n")

    def __call__(self, img, label) -> tuple:
        h, w = label.shape

        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if (pad_h > 0 or pad_w > 0) and self.padding:
            img = cv2.copyMakeBorder(
                img,
                top=pad_h_half,
                bottom=pad_h - pad_h_half,
                left=pad_w_half,
                right=pad_w - pad_w_half,
                borderType=cv2.BORDER_CONSTANT,
                value=self.padding,
            )
            label = cv2.copyMakeBorder(
                label,
                top=pad_h_half,
                bottom=pad_h - pad_h_half,
                left=pad_w_half,
                right=pad_w - pad_w_half,
                borderType=cv2.BORDER_CONSTANT,
                value=self.ignore_label,
                # value=0,
            )

        # update shape
        h, w = label.shape

        if self.crop_type == "rand":
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)

        img = img[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        label = label[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w]
        return img, label


@dataclass
class RandomRotate(object):
    rotate: tuple[int, int]
    padding: Optional[list[float]]
    ignore_label: int = 255
    p: float = 0.5

    def __call__(self, img, label) -> tuple:
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            img = cv2.warpAffine(
                img,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.padding,
            )
            label = cv2.warpAffine(
                label,
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.ignore_label,
                # borderValue=0,
            )
        return img, label


@dataclass
class RandomHorizontalFlip(object):
    p: float = 0.5

    def __call__(self, img, label) -> tuple:
        if random.random() < self.p:
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)
        return img, label


@dataclass
class RandomVerticalFlip(object):
    p: float = 0.5

    def __call__(self, img: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = cv2.flip(img, 0)
            label = cv2.flip(label, 0)
        return img, label


@dataclass
class RandomGaussianBlur(object):
    k_size: Union[int, list[int]] = 5
    p: float = 0.5
    __kernel_size: list[int] = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.k_size, int):
            self.__kernel_size = [self.k_size, self.k_size]

    def __call__(self, img: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = cv2.GaussianBlur(img, self.__kernel_size, 0)
        return img, label

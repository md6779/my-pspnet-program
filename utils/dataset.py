from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.preset import Preset


class CustomDatasetBase(Dataset):
    def __init__(
        self,
        root: Path,
        list_fp: Path,
        transform: Optional[Callable],
        phase: str = Preset.TRAIN,
    ) -> None:
        self.transform = transform
        self.data_list = self.make_dataset(root, list_fp, phase)

    def make_dataset(self, root: Path, list_fp: Path, phase: str = Preset.TRAIN) -> list:
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        raise NotImplementedError


class CustomDataset(CustomDatasetBase):
    def __init__(
        self,
        root: Path,
        list_fp: Path,
        transform: Optional[Callable],
        phase: str = Preset.TRAIN,
    ) -> None:
        super().__init__(root, list_fp, transform, phase)

    def make_dataset(self, root: Path, list_fp: Path, phase: str) -> list:
        label_dir = root / Preset.SEGMENTATION_PATH
        img_dir = root / Preset.IMAGE_PATH

        data_list: list[tuple[str, str]] = []
        for line in open(list_fp).readlines():
            stem = Path(line.strip())
            if stem.is_absolute():
                img_fp = stem.with_suffix(".jpg")
                label_fp = Path(str(stem).replace(Preset.IMAGE_PATH, Preset.SEGMENTATION_PATH)).with_suffix(".png")
            else:
                img_fp = img_dir / f"{stem}.jpg"
                label_fp = label_dir / f"{stem}.png"
            data_list.append((str(img_fp), str(label_fp)))

        print(f"Totally {len(data_list)} samples in {phase} set.")
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_fp, label_fp = self.data_list[index]
        img = np.array(Image.open(img_fp).convert("RGB"), dtype=np.float32)
        label = np.array(Image.open(label_fp).convert("P"))

        if self.transform:
            img, label = self.transform(img, label)
        return img, label


@dataclass(init=False)
class CustomDatasetWithoutAnnotations(CustomDatasetBase):
    def __init__(
        self,
        root: Path,
        list_fp: Path,
        transform: Optional[Callable],
        phase: str = Preset.TRAIN,
    ) -> None:
        super().__init__(root, list_fp, transform, phase)

    def make_dataset(self, root: Path, list_fp: Path, phase: str) -> list:
        img_dir = root / Preset.IMAGE_PATH

        data_list: list[tuple[str, str]] = []
        for line in open(list_fp).readlines():
            stem = line.strip()
            if Path(stem).is_absolute():
                img_fp = stem
            else:
                img_fp = img_dir / f"{stem}.jpg"
            data_list.append((str(img_fp), ""))

        print(f"Totally {len(data_list)} samples in {phase} set.")
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_fp, _ = self.data_list[index]
        img = np.array(Image.open(img_fp).convert("RGB"), dtype=np.float32)
        h, w, *_ = img.shape
        label = np.array(Image.new(mode="P", size=(h, w)), dtype=np.float32)

        if self.transform:
            img, label = self.transform(img, label)
        return img, label

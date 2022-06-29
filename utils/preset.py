from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    SEGMENTATION_PATH = "segmentations"
    IMAGE_PATH = "images"

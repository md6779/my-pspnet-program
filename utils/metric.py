from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Any, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@dataclass
class Metrics:
    SMOOTH: float = 1e-6

    main_loss_meter: AverageMeter = field(init=False, default=AverageMeter())
    aux_loss_meter: AverageMeter = field(init=False, default=AverageMeter())
    loss_meter: AverageMeter = field(init=False, default=AverageMeter())
    intersection_meter: AverageMeter = field(init=False, default=AverageMeter())
    union_meter: AverageMeter = field(init=False, default=AverageMeter())
    target_meter: AverageMeter = field(init=False, default=AverageMeter())

    @property
    def batch_loss(self) -> Any:
        return self.loss_meter.val

    @property
    def batch_main_loss(self) -> Any:
        return self.main_loss_meter.val

    @property
    def loss(self) -> Any:
        return self.main_loss_meter.avg

    @property
    def batch_acc(self) -> Union[Any, float]:
        return sum(self.intersection_meter.val) / (sum(self.target_meter.val) + self.SMOOTH)  # type: ignore

    @property
    def mean_batch_iou(self) -> Any:
        return np.mean(self.intersection_meter.val / (self.union_meter.val + self.SMOOTH))

    @property
    def mean_batch_acc(self) -> Any:
        return np.mean(self.intersection_meter.val / (self.target_meter.val + self.SMOOTH))

    @property
    def iou_class(self) -> Any:
        return self.intersection_meter.sum / (self.union_meter.sum + self.SMOOTH)

    @property
    def class_acc(self) -> Any:
        return self.intersection_meter.sum / (self.target_meter.sum + self.SMOOTH)

    @property
    def mean_iou(self) -> Any:
        return np.mean(self.iou_class)

    @property
    def mean_acc(self) -> Any:
        return np.mean(self.class_acc)

    @property
    def all_acc(self) -> Any:
        return sum(self.intersection_meter.sum) / (sum(self.target_meter.sum) + self.SMOOTH)  # type: ignore

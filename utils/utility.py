from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import torch
from termcolor import colored
from texttable import Texttable
from torch import Tensor
from torchvision.transforms import functional as F


def show_dict_as_pretty(d: dict[str, Any], indent: str = "") -> None:
    for k, v in d.items():
        if isinstance(v, dict):
            print(colored(f"{indent}{k}:", "cyan"))
            show_dict_as_pretty(v, indent=f"{indent}   ")
        else:
            print(f"{indent}{k}: {v}")
    print()


def get_texttable(header: list[str] = ["Key", "Value"], cols_align: str = "lr") -> Texttable:
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.header(header)
    table.set_cols_align([c for c in cols_align])
    return table


def subdivide_batch(
    x: Tensor,
    y: Tensor,
    device: torch.device,
    subdivisions: int,
    *,
    non_blocking: bool = False,
) -> Iterator[tuple[Tensor, Tensor]]:
    for a, b in zip(torch.chunk(x, subdivisions), torch.chunk(y, subdivisions)):
        yield (
            a.to(device=device, non_blocking=non_blocking),
            b.to(device=device, non_blocking=non_blocking),
        )


def display_image_as_pil(tensor: Tensor, palette: list[int]) -> None:
    tensor = tensor.to(torch.uint8)
    img = F.to_pil_image(tensor)
    img.putpalette(palette)
    img.show()


def print_img_array_in_console(tensor: Tensor, size: tuple[int, int] = (10, 10)) -> None:
    tensor = tensor.to(torch.uint8)
    img = F.to_pil_image(tensor).resize(size)
    for x in np.array(img):
        print(x)


def intersectionAndUnionGPU(
    output: Tensor, target: Tensor, K: int, ignore_index: int = 255
) -> tuple[Tensor, Tensor, Tensor]:
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnion(
    output: np.ndarray, target: np.ndarray, K: int, ignore_index: int = 255
) -> tuple[Any, Any, Any]:
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape

    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection  # type: ignore

    # ? * important: ignore background class
    # area_intersection = area_intersection[1:]
    # area_target = area_target[1:]
    # area_union = area_union[1:]
    # print(f"intersection = {area_intersection}")
    # print(f"union = {area_union}")
    # print(f"iou = {area_intersection / (area_target + 1e-6)}")  # type: ignore
    # exit()
    return area_intersection, area_union, area_target

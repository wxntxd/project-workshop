"""
Реализация torch.utils.data.Dataset поверх COCO-аннотаций.

Каждый элемент возвращает пару (image, target), где target — словарь
с полями boxes [N, 4] в формате [x1, y1, x2, y2], labels [N] и image_id,
то есть формат, ожидаемый детекторами torchvision (Faster R-CNN, SSD).
"""

import os

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from .transforms import get_transform


class KittiCocoDataset(Dataset):
    """Датасет KITTI, обёрнутый в интерфейс COCO."""

    def __init__(self, images_dir, ann_file, train=True):
        self.images_dir = images_dir
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())
        self.transform = get_transform(train)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image = Image.open(
            os.path.join(self.images_dir, img_info["file_name"])
        ).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])     # COCO xywh -> xyxy
            labels.append(ann["category_id"])

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }
        image, target = self.transform(image, target)
        return image, target


def collate_fn(batch):
    """Детекторы принимают списки изображений и целей переменной длины."""
    return tuple(zip(*batch))

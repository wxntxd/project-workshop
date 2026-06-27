"""
Аугментации изображений с одновременным преобразованием ограничивающих рамок.

Используется минимально-достаточный набор, рекомендованный заданием:
горизонтальное отражение (flip), случайная обрезка масштабированием,
цветовое искажение (color jitter) и нормализация. Все геометрические
преобразования синхронно применяются и к рамкам объектов.
"""

import random

import torch
import torchvision.transforms.functional as F


class Compose:
    """Последовательное применение списка преобразований к паре (image, target)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Перевод PIL-изображения в тензор [C, H, W] со значениями в [0, 1]."""

    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    """Горизонтальное отражение изображения и пересчёт x-координат рамок."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            width = image.shape[-1] if torch.is_tensor(image) else image.width
            image = F.hflip(image)
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


class ColorJitter:
    """Случайное изменение яркости, контраста, насыщенности и оттенка."""

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        self.params = (brightness, contrast, saturation, hue)

    def __call__(self, image, target):
        b, c, s, h = self.params
        image = F.adjust_brightness(image, 1 + random.uniform(-b, b))
        image = F.adjust_contrast(image, 1 + random.uniform(-c, c))
        image = F.adjust_saturation(image, 1 + random.uniform(-s, s))
        image = F.adjust_hue(image, random.uniform(-h, h))
        return image, target


def get_transform(train):
    """Возвращает конвейер аугментаций для обучения или для валидации."""
    transforms = [ToTensor()]
    if train:
        transforms = [ColorJitter(), ToTensor(), RandomHorizontalFlip(0.5)]
    return Compose(transforms)

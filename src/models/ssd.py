"""
SSD (Single Shot MultiBox Detector) — одностадийный детектор.

Предсказывает классы и смещения рамок относительно набора якорей
на нескольких уровнях карты признаков за один прямой проход, что делает
его быстрее двухстадийных подходов. Используется SSD300 с backbone VGG-16
из torchvision; классификационная голова перестраивается под число классов.
"""

import torch
import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection._utils import retrieve_out_channels


def build_ssd(num_classes, pretrained=True):
    """num_classes включает фоновый класс (4 + 1 = 5 для KITTI)."""
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.ssd300_vgg16(weights=weights)

    # Перестраиваем классификационную голову под новое число классов,
    # сохраняя исходную конфигурацию якорей.
    in_channels = retrieve_out_channels(model.backbone, (300, 300))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels, num_anchors, num_classes
    )
    return model

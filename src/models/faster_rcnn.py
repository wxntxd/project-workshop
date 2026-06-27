"""
Faster R-CNN — двухстадийный детектор.

Первая стадия (Region Proposal Network) генерирует кандидатные области,
вторая классифицирует их и уточняет рамки. Используется реализация
torchvision с предобученным на COCO backbone ResNet-50-FPN; заменяется
только классификационная "голова" под число классов KITTI.
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(num_classes, pretrained=True):
    """num_classes должно включать фоновый класс (для KITTI это 4 + 1 = 5)."""
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

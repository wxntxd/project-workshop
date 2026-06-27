"""
Реестр моделей. Скрывает различия в способах обучения за единым вызовом
train_model(name, cfg, logger) -> (history, metrics).

Faster R-CNN и SSD обучаются общим движком torchvision; YOLO, DETR
и EfficientDet имеют собственные обучающие циклы внутри своих модулей.
"""

from ..training.engine import run_torchvision
from .faster_rcnn import build_faster_rcnn
from .ssd import build_ssd

AVAILABLE_MODELS = ["yolo", "faster_rcnn", "ssd", "efficientdet", "detr"]


def train_model(name, cfg, logger):
    name = name.lower()
    if name == "faster_rcnn":
        return run_torchvision(cfg, logger, build_faster_rcnn, "Faster R-CNN")
    if name == "ssd":
        return run_torchvision(cfg, logger, build_ssd, "SSD")
    if name == "yolo":
        from . import yolo
        return yolo.run(cfg, logger)
    if name == "detr":
        from . import detr
        return detr.run(cfg, logger)
    if name == "efficientdet":
        from . import efficientdet
        return efficientdet.run(cfg, logger)
    raise ValueError(f"Неизвестная модель '{name}'. Доступны: {AVAILABLE_MODELS}")

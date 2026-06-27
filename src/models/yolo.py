"""
YOLO (You Only Look Once), версия v8 — одностадийный детектор.

Делит изображение на сетку и за один проход предсказывает рамки, классы
и оценки уверенности; современные версии бесякорные и оптимизированы под
скорость, что важно для бортовых систем автономного транспорта.
Используется реализация ultralytics (YOLOv8n).
"""

import os

from ..evaluation.metrics import unify_metrics


def run(cfg, logger):
    """Обучение и оценка YOLOv8. Возвращает (history, metrics).

    Библиотека ultralytics инкапсулирует обучающий цикл, логирование,
    расчёт mAP (формат COCO) и построение графиков. Мы лишь приводим
    итоговые метрики к единому виду для сравнения с другими моделями.
    """
    from ultralytics import YOLO

    weights = "yolov8n.pt"            # предобучено на COCO; n — самый лёгкий вариант
    model = YOLO(weights)

    results = model.train(
        data=cfg["data"]["yolo_yaml"],
        epochs=cfg["train"]["epochs"],
        imgsz=cfg["train"]["img_size"],
        batch=cfg["train"]["batch_size"],
        lr0=cfg["train"]["lr"] if cfg["train"]["lr"] > 1e-3 else 0.01,
        optimizer="AdamW" if cfg["train"]["optimizer"] == "adamw" else "SGD",
        project=cfg["output"]["results_dir"],
        name="yolov8_kitti",
        seed=cfg["seed"],
        device=0 if cfg["train"]["device"] == "cuda" else "cpu",
        verbose=False,
    )

    # Финальная валидация для извлечения метрик
    val_metrics = model.val(data=cfg["data"]["yolo_yaml"], verbose=False)

    metrics = unify_metrics(
        map50_95=float(val_metrics.box.map),
        map50=float(val_metrics.box.map50),
        precision=float(val_metrics.box.mp),
        recall=float(val_metrics.box.mr),
    )

    # ultralytics пишет историю обучения в results.csv внутри каталога эксперимента
    history = _read_history(os.path.join(cfg["output"]["results_dir"], "yolov8_kitti"))
    logger.info(f"[YOLOv8] mAP@0.5={metrics['mAP@0.5']:.4f} mAP@[.5:.95]={metrics['mAP@0.5:0.95']:.4f}")
    return history, metrics


def _read_history(run_dir):
    """Считывает кривые обучения из results.csv, сохранённого ultralytics."""
    csv_path = os.path.join(run_dir, "results.csv")
    history = {"train_loss": []}
    if not os.path.exists(csv_path):
        return history
    import csv

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v for k, v in row.items()}
            box = float(row.get("train/box_loss", 0))
            cls = float(row.get("train/cls_loss", 0))
            dfl = float(row.get("train/dfl_loss", 0))
            history["train_loss"].append(box + cls + dfl)
    return history

"""
Единый модуль оценки качества детекции.

Все модели приводятся к формату предсказаний COCO
(список словарей image_id / category_id / bbox[xywh] / score), после чего
метрики считаются одной и той же реализацией pycocotools. Это гарантирует
корректность сравнения моделей между собой.

Возвращаемые метрики:
  * mAP@0.5        — средняя точность при пороге IoU = 0.5;
  * mAP@0.5:0.95   — усреднение по порогам IoU от 0.5 до 0.95 (основная метрика COCO);
  * Precision      — средняя точность при IoU = 0.5;
  * Recall         — средняя полнота (AR@100) при IoU = 0.5;
  * F1             — гармоническое среднее Precision и Recall.
"""

import contextlib
import io

import numpy as np


def unify_metrics(map50_95, map50, precision, recall):
    """Сводит набор чисел к стандартному словарю метрик с расчётом F1."""
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "mAP@0.5": round(float(map50), 4),
        "mAP@0.5:0.95": round(float(map50_95), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1": round(float(f1), 4),
    }


def evaluate_coco(coco_gt, predictions):
    """Считает метрики по предсказаниям в формате COCO.

    coco_gt      — объект pycocotools.COCO с эталонной разметкой валидации;
    predictions  — список словарей предсказаний (может быть пустым).
    """
    from pycocotools.cocoeval import COCOeval

    if len(predictions) == 0:
        return unify_metrics(0.0, 0.0, 0.0, 0.0)

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    # Подавляем подробный вывод pycocotools, оставляя только нужные числа
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    map50_95 = coco_eval.stats[0]        # AP @[.5:.95]
    map50 = coco_eval.stats[1]           # AP @.5

    # Precision при IoU = 0.5: индекс 0 по оси порогов IoU,
    # последняя ось — maxDets=100, area=all (индекс 0).
    prec = coco_eval.eval["precision"][0, :, :, 0, -1]
    prec = prec[prec > -1]
    precision = float(prec.mean()) if prec.size else 0.0

    # Recall (AR@100) при IoU = 0.5.
    rec = coco_eval.eval["recall"][0, :, 0, -1]
    rec = rec[rec > -1]
    recall = float(rec.mean()) if rec.size else 0.0

    return unify_metrics(map50_95, map50, precision, recall)

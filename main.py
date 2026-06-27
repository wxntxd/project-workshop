"""
Точка входа проекта сравнения детекторов на датасете KITTI.

Примеры запуска:
    python main.py --prepare-data                 # конвертация KITTI -> COCO/YOLO
    python main.py --model yolo                    # обучить одну модель
    python main.py --model all --epochs 15         # обучить все 5 моделей
    python main.py --model faster_rcnn --lr 0.001  # переопределить гиперпараметр
"""

import argparse
import os

import yaml

from src.dataset import kitti
from src.models.registry import AVAILABLE_MODELS, train_model
from src.utils.utils import set_seed, get_logger, save_json
from src.utils import visualize


def load_config(path, overrides):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Переопределение гиперпараметров из командной строки
    if overrides.get("epochs") is not None:
        cfg["train"]["epochs"] = overrides["epochs"]
    if overrides.get("lr") is not None:
        cfg["train"]["lr"] = overrides["lr"]
    if overrides.get("batch_size") is not None:
        cfg["train"]["batch_size"] = overrides["batch_size"]
    if overrides.get("device") is not None:
        cfg["train"]["device"] = overrides["device"]
    return cfg


def build_comparison_table(all_metrics):
    """Формирует текстовую таблицу метрик всех моделей для отчёта."""
    cols = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]
    header = "| Модель | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [header, sep]
    for name, m in all_metrics.items():
        rows.append("| " + name + " | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Сравнение детекторов на KITTI")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="all",
                        help="имя модели или 'all' для всех пяти")
    parser.add_argument("--prepare-data", action="store_true",
                        help="выполнить конвертацию KITTI в форматы COCO/YOLO и выйти")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, vars(args))
    set_seed(cfg["seed"])
    logger = get_logger("experiment", cfg["output"]["logs_dir"])

    # Этап подготовки данных
    if args.prepare_data:
        kitti.convert(cfg["data"]["raw_dir"], cfg["data"]["processed_dir"],
                      cfg["data"]["val_fraction"], cfg["seed"])
        logger.info("Подготовка данных завершена.")
        return

    if not os.path.exists(cfg["data"]["coco_val_ann"]):
        logger.info("Аннотации не найдены — запускаю конвертацию данных.")
        kitti.convert(cfg["data"]["raw_dir"], cfg["data"]["processed_dir"],
                      cfg["data"]["val_fraction"], cfg["seed"])

    models = AVAILABLE_MODELS if args.model == "all" else [args.model]

    all_metrics, histories = {}, {}
    for name in models:
        logger.info(f"===== Запуск модели: {name} =====")
        try:
            history, metrics = train_model(name, cfg, logger)
        except Exception as exc:                      # одна модель не должна валить весь прогон
            logger.error(f"Модель {name} завершилась с ошибкой: {exc}")
            continue
        all_metrics[name] = metrics
        histories[name] = history
        visualize.plot_loss_curve(history, name, cfg["output"]["plots_dir"])
        save_json(metrics, os.path.join(cfg["output"]["results_dir"], f"metrics_{name}.json"))

    # Сводное сравнение
    if all_metrics:
        save_json(all_metrics, os.path.join(cfg["output"]["results_dir"], "all_metrics.json"))
        visualize.plot_all_losses(histories, cfg["output"]["plots_dir"])
        for metric in ("mAP@0.5", "mAP@0.5:0.95", "F1"):
            visualize.plot_comparison(all_metrics, cfg["output"]["plots_dir"], metric)
        table = build_comparison_table(all_metrics)
        with open(os.path.join(cfg["output"]["results_dir"], "comparison_table.md"), "w",
                  encoding="utf-8") as f:
            f.write(table)
        logger.info("Итоговая таблица сравнения:\n" + table)


if __name__ == "__main__":
    main()

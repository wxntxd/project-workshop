"""
Парсинг датасета KITTI (2D Object Detection) и его конвертация
в два целевых формата:
  * COCO JSON  — для torchvision (Faster R-CNN, SSD), DETR и EfficientDet;
  * YOLO txt   — для библиотеки ultralytics (YOLOv8).

Оригинальная разметка KITTI хранится в виде текстовых файлов label_2/<id>.txt,
каждая строка которых описывает один объект:

    type truncated occluded alpha  x1 y1 x2 y2  h w l  X Y Z  rotation_y

Для задачи 2D-детекции используются только поля type и (x1, y1, x2, y2).
"""

import json
import os
import random
from collections import Counter
from pathlib import Path

from PIL import Image

# Сведение исходных 8 значимых меток KITTI к 4 рабочим классам проекта.
# Категории DontCare и Misc исключаются как неинформативные.
KITTI_TO_CLASS = {
    "Car": "car",
    "Van": "car",
    "Truck": "truck",
    "Tram": "truck",
    "Pedestrian": "pedestrian",
    "Person_sitting": "pedestrian",
    "Cyclist": "cyclist",
}

# Порядок классов фиксирован: индекс в списке = id класса в YOLO,
# id категории в COCO = индекс + 1 (COCO нумерует категории с единицы).
CLASS_NAMES = ["car", "pedestrian", "cyclist", "truck"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def parse_label_file(label_path):
    """Считывает один файл разметки KITTI и возвращает список объектов.

    Каждый объект — словарь с именем класса и координатами рамки в формате
    [x1, y1, x2, y2] (левый-верхний и правый-нижний угол в пикселях).
    Объекты неиспользуемых классов отбрасываются.
    """
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            kitti_type = parts[0]
            if kitti_type not in KITTI_TO_CLASS:
                continue
            cls_name = KITTI_TO_CLASS[kitti_type]
            x1, y1, x2, y2 = map(float, parts[4:8])
            # Отбрасываем вырожденные рамки нулевой площади
            if x2 <= x1 or y2 <= y1:
                continue
            objects.append({"class": cls_name, "bbox": [x1, y1, x2, y2]})
    return objects


def split_ids(image_ids, val_fraction, seed):
    """Детерминированное разбиение списка id на train/val по заданной доле."""
    ids = sorted(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    val_ids = set(ids[:n_val])
    train_ids = [i for i in ids if i not in val_ids]
    val_ids = [i for i in ids if i in val_ids]
    return train_ids, val_ids


def _build_coco_dict(image_ids, images_dir, labels_dir):
    """Формирует словарь в формате COCO для подмножества изображений."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": CLASS_TO_ID[name] + 1, "name": name}
            for name in CLASS_NAMES
        ],
    }
    ann_id = 1
    for img_idx, image_id in enumerate(image_ids, start=1):
        img_file = os.path.join(images_dir, f"{image_id}.png")
        width, height = Image.open(img_file).size
        coco["images"].append(
            {
                "id": img_idx,
                "file_name": f"{image_id}.png",
                "width": width,
                "height": height,
            }
        )
        label_file = os.path.join(labels_dir, f"{image_id}.txt")
        if not os.path.exists(label_file):
            continue
        for obj in parse_label_file(label_file):
            x1, y1, x2, y2 = obj["bbox"]
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": CLASS_TO_ID[obj["class"]] + 1,
                    "bbox": [x1, y1, w, h],          # COCO: [x, y, ширина, высота]
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return coco


def _write_yolo_labels(image_ids, images_dir, labels_dir, out_dir):
    """Создаёт YOLO-разметку: для каждого изображения файл со строками
    `class_id cx cy w h`, где координаты нормированы в диапазон [0, 1].
    """
    os.makedirs(out_dir, exist_ok=True)
    for image_id in image_ids:
        img_file = os.path.join(images_dir, f"{image_id}.png")
        width, height = Image.open(img_file).size
        label_file = os.path.join(labels_dir, f"{image_id}.txt")
        lines = []
        if os.path.exists(label_file):
            for obj in parse_label_file(label_file):
                x1, y1, x2, y2 = obj["bbox"]
                cx = (x1 + x2) / 2 / width
                cy = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                lines.append(
                    f"{CLASS_TO_ID[obj['class']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
        with open(os.path.join(out_dir, f"{image_id}.txt"), "w") as f:
            f.write("\n".join(lines))


def convert(raw_dir, processed_dir, val_fraction=0.2, seed=42):
    """Главная процедура конвертации KITTI -> COCO + YOLO.

    Ожидаемая структура raw_dir:
        raw_dir/training/image_2/*.png
        raw_dir/training/label_2/*.txt
    """
    images_dir = os.path.join(raw_dir, "training", "image_2")
    labels_dir = os.path.join(raw_dir, "training", "label_2")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Не найден каталог изображений: {images_dir}")

    image_ids = [Path(p).stem for p in os.listdir(images_dir) if p.endswith(".png")]
    train_ids, val_ids = split_ids(image_ids, val_fraction, seed)
    print(f"Всего изображений: {len(image_ids)} | train: {len(train_ids)} | val: {len(val_ids)}")

    # --- COCO ---
    coco_dir = os.path.join(processed_dir, "coco")
    os.makedirs(coco_dir, exist_ok=True)
    for split, ids in (("train", train_ids), ("val", val_ids)):
        coco = _build_coco_dict(ids, images_dir, labels_dir)
        out_path = os.path.join(coco_dir, f"instances_{split}.json")
        with open(out_path, "w") as f:
            json.dump(coco, f)
        cnt = Counter(a["category_id"] for a in coco["annotations"])
        dist = {CLASS_NAMES[k - 1]: v for k, v in sorted(cnt.items())}
        print(f"[COCO] {split}: {len(coco['annotations'])} объектов, распределение {dist}")

    # --- YOLO ---
    yolo_dir = os.path.join(processed_dir, "yolo")
    for split, ids in (("train", train_ids), ("val", val_ids)):
        _write_yolo_labels(ids, images_dir, labels_dir,
                           os.path.join(yolo_dir, "labels", split))
        # YOLO ожидает изображения рядом; создаём символические ссылки
        img_link_dir = os.path.join(yolo_dir, "images", split)
        os.makedirs(img_link_dir, exist_ok=True)
        for image_id in ids:
            src = os.path.abspath(os.path.join(images_dir, f"{image_id}.png"))
            dst = os.path.join(img_link_dir, f"{image_id}.png")
            if not os.path.exists(dst):
                os.symlink(src, dst)

    # Файл-описание датасета для ultralytics
    yaml_path = os.path.join(yolo_dir, "kitti.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(yolo_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")
    print(f"[YOLO] конфигурация записана в {yaml_path}")
    return train_ids, val_ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Конвертация KITTI в COCO и YOLO")
    parser.add_argument("--raw_dir", default="data/raw/kitti")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert(args.raw_dir, args.processed_dir, args.val_fraction, args.seed)

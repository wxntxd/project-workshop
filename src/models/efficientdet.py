"""
EfficientDet — одностадийный детектор с эффективным масштабированием.

Backbone EfficientNet извлекает признаки, взвешенная двунаправленная
пирамида признаков (BiFPN) многократно сливает их на разных масштабах,
общая голова предсказывает классы и рамки. Реализация — библиотека effdet
(вариант tf_efficientdet_d0) с предобученным backbone.

Замечание о воспроизводимости: библиотека effdet требует фиксированного
квадратного размера входа и специфического формата целей (bbox в порядке
ymin, xmin, ymax, xmax). Этот модуль инкапсулирует все преобразования.
"""

import os

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

from ..dataset.kitti import CLASS_NAMES
from ..evaluation.metrics import evaluate_coco


class EffDetDataset(Dataset):
    """Готовит изображения фиксированного размера и цели в формате effdet."""

    def __init__(self, images_dir, ann_file, img_size=512, train=True):
        from pycocotools.coco import COCO

        self.images_dir = images_dir
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(self.images_dir, info["file_name"])).convert("RGB")
        w0, h0 = image.size
        scale = self.img_size / max(w0, h0)
        nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
        image = image.resize((nw, nh))

        # Дополняем до квадрата img_size x img_size справа и снизу
        tensor = TF.to_tensor(image)
        padded = torch.zeros(3, self.img_size, self.img_size)
        padded[:, :nh, :nw] = tensor

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes, labels = [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            x1, y1, x2, y2 = x * scale, y * scale, (x + bw) * scale, (y + bh) * scale
            boxes.append([y1, x1, y2, x2])           # effdet ожидает ymin,xmin,ymax,xmax
            labels.append(ann["category_id"])         # уже 1-индексированы
        if not boxes:
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [1]

        target = {
            "bbox": torch.tensor(boxes, dtype=torch.float32),
            "cls": torch.tensor(labels, dtype=torch.float32),
            "img_scale": torch.tensor([1.0 / scale]),
            "img_size": torch.tensor([(h0, w0)], dtype=torch.float32),
            "img_id": img_id,
        }
        return padded, target


def _collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {
        "bbox": [b[1]["bbox"] for b in batch],
        "cls": [b[1]["cls"] for b in batch],
        "img_scale": torch.cat([b[1]["img_scale"] for b in batch]),
        "img_size": torch.cat([b[1]["img_size"] for b in batch]),
        "img_id": [b[1]["img_id"] for b in batch],
    }
    return images, targets


def _build_model(img_size):
    from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
    from effdet.efficientdet import HeadNet

    config = get_efficientdet_config("tf_efficientdet_d0")
    config.num_classes = len(CLASS_NAMES)
    config.image_size = [img_size, img_size]
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    return DetBenchTrain(net, config), DetBenchPredict(net), net


def run(cfg, logger):
    """Полный цикл обучения и оценки EfficientDet. Возвращает (history, metrics)."""
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    img_size = 512
    bench_train, bench_predict, _ = _build_model(img_size)
    bench_train.to(device)

    images_dir = cfg["data"]["images_dir"]
    train_ds = EffDetDataset(images_dir, cfg["data"]["coco_train_ann"], img_size, train=True)
    val_ds = EffDetDataset(images_dir, cfg["data"]["coco_val_ann"], img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              collate_fn=_collate, num_workers=cfg["train"]["num_workers"])

    optimizer = torch.optim.AdamW(bench_train.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])

    history = {"train_loss": []}
    for epoch in range(cfg["train"]["epochs"]):
        bench_train.train()
        running = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            tgt = {
                "bbox": [b.to(device) for b in targets["bbox"]],
                "cls": [c.to(device) for c in targets["cls"]],
                "img_scale": targets["img_scale"].to(device),
                "img_size": targets["img_size"].to(device),
            }
            optimizer.zero_grad()
            out = bench_train(images, tgt)
            out["loss"].backward()
            optimizer.step()
            running += out["loss"].item()
        epoch_loss = running / max(1, len(train_loader))
        history["train_loss"].append(epoch_loss)
        logger.info(f"[EfficientDet] epoch {epoch + 1}/{cfg['train']['epochs']} loss={epoch_loss:.4f}")

    metrics = _evaluate(bench_predict.to(device), val_ds, cfg, device)
    return history, metrics


@torch.no_grad()
def _evaluate(bench_predict, val_ds, cfg, device):
    bench_predict.eval()
    loader = DataLoader(val_ds, batch_size=1, collate_fn=_collate)
    predictions = []
    for images, targets in loader:
        images = images.to(device)
        det = bench_predict(images, img_scales=targets["img_scale"].to(device),
                            img_size=targets["img_size"].to(device))[0]
        img_id = targets["img_id"][0]
        for row in det.cpu().numpy():
            x1, y1, x2, y2, score, cls = row
            if score < cfg["eval"]["score_threshold"]:
                continue
            predictions.append({
                "image_id": img_id,
                "category_id": int(cls),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })
    return evaluate_coco(val_ds.coco, predictions)

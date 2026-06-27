"""
DETR (DEtection TRansformer) — детектор на основе трансформера.

Свёрточный backbone извлекает признаки, кодер-декодер трансформера
обрабатывает их через механизм внимания, а фиксированный набор
обучаемых "object queries" напрямую предсказывает множество объектов
без якорей и без постобработки NMS. Используется facebook/detr-resnet-50.
"""

import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

from ..dataset.kitti import CLASS_NAMES
from ..evaluation.metrics import evaluate_coco


class DetrDataset(Dataset):
    """Готовит батчи в формате, ожидаемом DetrImageProcessor."""

    def __init__(self, images_dir, ann_file, processor):
        from pycocotools.coco import COCO

        self.images_dir = images_dir
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())
        self.processor = processor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        from PIL import Image

        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(self.images_dir, info["file_name"])).convert("RGB")
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        target = {"image_id": img_id, "annotations": anns}
        enc = self.processor(images=image, annotations=target, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "labels": enc["labels"][0],
            "orig_id": img_id,
        }


def _collate(processor):
    def fn(batch):
        pixel_values = [b["pixel_values"] for b in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": [b["labels"] for b in batch],
            "orig_ids": [b["orig_id"] for b in batch],
        }
    return fn


def run(cfg, logger):
    """Полный цикл обучения и оценки DETR. Возвращает (history, metrics)."""
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True,
    ).to(device)

    images_dir = cfg["data"]["images_dir"]
    train_ds = DetrDataset(images_dir, cfg["data"]["coco_train_ann"], processor)
    val_ds = DetrDataset(images_dir, cfg["data"]["coco_val_ann"], processor)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, collate_fn=_collate(processor),
                              num_workers=cfg["train"]["num_workers"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])

    history = {"train_loss": []}
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                pixel_mask=batch["pixel_mask"].to(device),
                labels=[{k: v.to(device) for k, v in t.items()} for t in batch["labels"]],
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            running += outputs.loss.item()
        epoch_loss = running / max(1, len(train_loader))
        history["train_loss"].append(epoch_loss)
        logger.info(f"[DETR] epoch {epoch + 1}/{cfg['train']['epochs']} loss={epoch_loss:.4f}")

    metrics = _evaluate(model, processor, val_ds, cfg, device)
    return history, metrics


@torch.no_grad()
def _evaluate(model, processor, val_ds, cfg, device):
    """Прогон по валидации и расчёт mAP в формате COCO."""
    model.eval()
    predictions = []
    loader = DataLoader(val_ds, batch_size=1, collate_fn=_collate(processor))
    for batch in loader:
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            pixel_mask=batch["pixel_mask"].to(device),
        )
        img_id = batch["orig_ids"][0]
        h = val_ds.coco.loadImgs(img_id)[0]["height"]
        w = val_ds.coco.loadImgs(img_id)[0]["width"]
        results = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([[h, w]]),
            threshold=cfg["eval"]["score_threshold"],
        )[0]
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            predictions.append({
                "image_id": img_id,
                "category_id": int(label) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    return evaluate_coco(val_ds.coco, predictions)

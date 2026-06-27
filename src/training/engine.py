"""
Универсальный цикл обучения и оценки для детекторов torchvision
(Faster R-CNN и SSD), имеющих единый программный интерфейс:
в режиме обучения модель принимает (images, targets) и возвращает словарь
функций потерь, в режиме инференса — список предсказаний на изображение.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset.dataset import KittiCocoDataset, collate_fn
from ..evaluation.metrics import evaluate_coco


def _build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["train"]["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["train"]["lr"],
                               momentum=cfg["train"]["momentum"],
                               weight_decay=cfg["train"]["weight_decay"])
    return torch.optim.AdamW(params, lr=cfg["train"]["lr"],
                             weight_decay=cfg["train"]["weight_decay"])


def _build_scheduler(optimizer, cfg):
    name = cfg["train"]["lr_scheduler"]
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg["train"]["epochs"] // 3), gamma=0.1)
    return None


def train_one_epoch(model, optimizer, loader, device, scaler, epoch, logger):
    """Один проход по обучающей выборке. Возвращает средний loss за эпоху."""
    model.train()
    running = 0.0
    for images, targets in tqdm(loader, desc=f"epoch {epoch + 1}", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, val_ds, cfg, device):
    """Прогон по валидации и расчёт метрик в формате COCO."""
    model.eval()
    loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn,
                        num_workers=cfg["train"]["num_workers"])
    predictions = []
    score_thr = cfg["eval"]["score_threshold"]
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        img_id = int(targets[0]["image_id"].item())
        out = outputs[0]
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            if score < score_thr:
                continue
            x1, y1, x2, y2 = box.tolist()
            predictions.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    return evaluate_coco(val_ds.coco, predictions)


def run_torchvision(cfg, logger, build_fn, model_name):
    """Полный цикл для модели torchvision. Возвращает (history, metrics)."""
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    num_classes = len(cfg["data"]["classes"]) + 1     # + фоновый класс
    model = build_fn(num_classes).to(device)

    images_dir = cfg["data"]["images_dir"]
    train_ds = KittiCocoDataset(images_dir, cfg["data"]["coco_train_ann"], train=True)
    val_ds = KittiCocoDataset(images_dir, cfg["data"]["coco_val_ann"], train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              collate_fn=collate_fn, num_workers=cfg["train"]["num_workers"])

    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    history = {"train_loss": []}
    for epoch in range(cfg["train"]["epochs"]):
        loss = train_one_epoch(model, optimizer, train_loader, device, scaler, epoch, logger)
        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(loss)
        logger.info(f"[{model_name}] epoch {epoch + 1}/{cfg['train']['epochs']} loss={loss:.4f}")

    metrics = evaluate(model, val_ds, cfg, device)
    logger.info(f"[{model_name}] mAP@0.5={metrics['mAP@0.5']:.4f} "
                f"mAP@[.5:.95]={metrics['mAP@0.5:0.95']:.4f}")
    return history, metrics

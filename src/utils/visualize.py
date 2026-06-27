"""Визуализация: кривые обучения, сравнительные диаграммы, предсказания на изображениях."""

import os

import matplotlib.pyplot as plt


def plot_loss_curve(history, model_name, out_dir):
    """Строит кривую функции потерь по эпохам для одной модели."""
    losses = history.get("train_loss", [])
    if not losses:
        return None
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Эпоха")
    plt.ylabel("Функция потерь (train)")
    plt.title(f"Кривая обучения: {model_name}")
    plt.grid(True, alpha=0.3)
    path = os.path.join(out_dir, f"loss_{model_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_comparison(all_metrics, out_dir, metric="mAP@0.5"):
    """Столбчатая диаграмма сравнения моделей по выбранной метрике."""
    os.makedirs(out_dir, exist_ok=True)
    names = list(all_metrics.keys())
    values = [all_metrics[n][metric] for n in names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color="#4C72B0")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                 ha="center", va="bottom")
    plt.ylabel(metric)
    plt.title(f"Сравнение моделей по метрике {metric}")
    plt.xticks(rotation=20)
    plt.grid(True, axis="y", alpha=0.3)
    path = os.path.join(out_dir, f"comparison_{metric.replace('@', '').replace(':', '_')}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_all_losses(histories, out_dir):
    """Накладывает кривые потерь всех моделей на один график."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for name, hist in histories.items():
        losses = hist.get("train_loss", [])
        if losses:
            plt.plot(range(1, len(losses) + 1), losses, marker="o", label=name)
    plt.xlabel("Эпоха")
    plt.ylabel("Функция потерь (train)")
    plt.title("Сравнение кривых обучения")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "all_losses.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def draw_predictions(image, boxes, labels, scores, class_names, out_path, score_thr=0.5):
    """Отрисовывает предсказанные рамки на изображении и сохраняет результат."""
    import numpy as np

    plt.figure(figsize=(10, 6))
    plt.imshow(np.array(image))
    ax = plt.gca()
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, color="red", linewidth=2))
        name = class_names[int(label) - 1] if 0 < int(label) <= len(class_names) else str(label)
        ax.text(x1, y1 - 4, f"{name} {score:.2f}", color="white",
                fontsize=9, bbox=dict(facecolor="red", alpha=0.7, pad=1))
    plt.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path

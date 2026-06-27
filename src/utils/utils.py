"""Вспомогательные функции: фиксация случайности, логирование, ввод-вывод."""

import json
import logging
import os
import random

import numpy as np


def set_seed(seed):
    """Фиксирует генераторы случайных чисел для воспроизводимости экспериментов."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Детерминированный режим cudnn (ценой небольшого замедления)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def get_logger(name, log_dir):
    """Создаёт логгер, пишущий одновременно в консоль и в файл."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

"""Тонкая обёртка над src.dataset.kitti.convert для запуска из каталога scripts."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset.kitti import convert

if __name__ == "__main__":
    convert("data/raw/kitti", "data/processed", val_fraction=0.2, seed=42)

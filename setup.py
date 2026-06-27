from setuptools import setup, find_packages

setup(
    name="cv-kitti-detection",
    version="0.1.0",
    description="Сравнение детекторов объектов (YOLO, Faster R-CNN, SSD, EfficientDet, DETR) на датасете KITTI",
    author="Mamedov Amal, BVT2401, MTUCI",
    packages=find_packages(),
    python_requires=">=3.9",
)

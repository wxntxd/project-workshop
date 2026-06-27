#!/usr/bin/env bash
# Загрузка датасета KITTI 2D Object Detection.
#
# Внимание: для скачивания требуется бесплатная регистрация на сайте KITTI
# (https://www.cvlibs.net/datasets/kitti/eval_object.php). После регистрации
# скачайте два архива и поместите их в data/raw/, затем запустите этот скрипт:
#   * data_object_image_2.zip  (левые цветные изображения, ~12 ГБ)
#   * data_object_label_2.zip  (разметка обучающей выборки)
#
# Альтернатива: тот же датасет доступен как "KITTI" на Kaggle и Roboflow,
# что удобно при работе из Google Colab.
set -e

RAW_DIR="data/raw/kitti"
mkdir -p "${RAW_DIR}"

if [ -f "data/raw/data_object_image_2.zip" ]; then
    echo "Распаковка изображений..."
    unzip -q -o data/raw/data_object_image_2.zip -d "${RAW_DIR}"
fi
if [ -f "data/raw/data_object_label_2.zip" ]; then
    echo "Распаковка разметки..."
    unzip -q -o data/raw/data_object_label_2.zip -d "${RAW_DIR}"
fi

echo "Ожидаемая структура:"
echo "  ${RAW_DIR}/training/image_2/*.png"
echo "  ${RAW_DIR}/training/label_2/*.txt"

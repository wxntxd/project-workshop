# Сравнение детекторов объектов на датасете KITTI

Учебная практика (технологическая). Проект полного цикла в области
компьютерного зрения: обучение и сравнение **пяти** современных детекторов
(YOLOv8, Faster R-CNN, SSD, EfficientDet, DETR) на задаче обнаружения
объектов дорожной сцены для систем автономного транспорта.

**Автор:** Мамедов Амаль, группа БВТ2401, МТУСИ, кафедра МКиИТ.

## Решаемая задача

Обнаружение и локализация четырёх классов объектов дорожной сцены:
`car`, `pedestrian`, `cyclist`, `truck`. Вход — RGB-изображение,
выход — набор ограничивающих рамок с классами и оценками уверенности.
Основная метрика качества — **mAP@0.5** и **mAP@[0.5:0.95]** (формат COCO).

## Структура репозитория

```
cv-kitti-detection/
├── configs/default.yaml          # гиперпараметры всех экспериментов
├── data/
│   ├── raw/                       # оригинальный KITTI (не версионируется)
│   └── processed/                 # сгенерированные COCO- и YOLO-аннотации
├── src/
│   ├── dataset/                   # парсинг KITTI, конвертация, аугментации, Dataset
│   ├── models/                    # 5 детекторов + реестр
│   ├── training/                  # общий движок обучения torchvision-моделей
│   ├── evaluation/                # единый расчёт mAP (pycocotools)
│   └── utils/                     # seed, логирование, графики
├── notebooks/exploration.ipynb    # разведочный анализ данных
├── results/                       # метрики, графики, логи (генерируются)
├── scripts/                       # загрузка и конвертация данных
└── main.py                        # точка входа
```

## Установка

```bash
git clone <repository_url>
cd cv-kitti-detection
pip install -r requirements.txt
```

Рекомендуется GPU (например, бесплатный Google Colab с T4). Без GPU обучение
возможно, но крайне медленное.

## Подготовка данных

1. Зарегистрируйтесь на сайте KITTI и скачайте архивы
   `data_object_image_2.zip` и `data_object_label_2.zip` в `data/raw/`.
2. Распакуйте и сконвертируйте:

```bash
bash scripts/download_kitti.sh
python main.py --prepare-data
```

После этого появятся `data/processed/coco/instances_{train,val}.json`
и YOLO-разметка с файлом `kitti.yaml`.

## Запуск экспериментов

```bash
# одна модель
python main.py --model yolo
python main.py --model faster_rcnn --epochs 15 --lr 0.0005

# все пять моделей и автоматическое сравнение
python main.py --model all --epochs 15
```

Результаты сохраняются в `results/`:
- `metrics_<model>.json` и `all_metrics.json` — числовые метрики;
- `comparison_table.md` — сводная таблица;
- `plots/` — кривые потерь и столбчатые сравнения моделей.

## Доступные модели

| Имя для `--model` | Архитектура | Библиотека |
|---|---|---|
| `yolo`         | YOLOv8n (одностадийный, бесякорный) | ultralytics |
| `faster_rcnn`  | Faster R-CNN (двухстадийный) | torchvision |
| `ssd`          | SSD300-VGG16 (одностадийный) | torchvision |
| `efficientdet` | EfficientDet-D0 (BiFPN) | effdet |
| `detr`         | DETR-ResNet50 (трансформер) | transformers |

## Воспроизводимость

Случайность зафиксирована (`seed: 42` в конфиге), детерминированный режим
cuDNN включён, все гиперпараметры вынесены в `configs/default.yaml`.

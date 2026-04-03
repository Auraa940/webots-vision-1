#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_rmc1_detector.py

Что делает:
1) Проверяет сырой датасет:
   raw_dataset/
     images/*.jpg|png|jpeg...
     labels/*.txt   (YOLO format: class x_center y_center width height)
2) Делит его на train/val/test
3) Собирает YOLO-структуру
4) Обучает модель Ultralytics YOLO
5) Валидирует
6) Экспортирует best.pt -> ONNX

Пример запуска:
python3 train_rmc1_detector.py \
  --raw-dataset /home/user/datasets/rmc1_raw \
  --prepared-dataset /home/user/datasets/rmc1_prepared \
  --project /home/user/runs/rmc1 \
  --name yolo_boxes \
  --model yolov8n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --device cpu


Структура сырого датасета для этого скрипта
rmc1_raw/
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── labels/
    ├── 0001.txt
    ├── 0002.txt
    └── ...

Пример 0001.txt:
0 0.512 0.481 0.221 0.198
2 0.231 0.655 0.205 0.190

"""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import yaml
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dataset", type=str, required=True,
                        help="Путь к сырому датасету: images/ и labels/")
    parser.add_argument("--prepared-dataset", type=str, required=True,
                        help="Куда собрать train/val/test структуру")
    parser.add_argument("--project", type=str, required=True,
                        help="Папка для результатов обучения")
    parser.add_argument("--name", type=str, default="rmc1_detector",
                        help="Имя запуска")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Базовая модель YOLO для дообучения")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu / 0 / 0,1 ...")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument("--classes", type=str, default="hammer,wrench,pliers",
                        help="Список классов через запятую")

    # Лёгкие аугментации
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.10)
    parser.add_argument("--fliplr", type=float, default=0.5)

    parser.add_argument("--export-onnx", action="store_true",
                        help="Экспортировать best.onnx после обучения")
    parser.add_argument("--onnx-opset", type=int, default=12)
    parser.add_argument("--simplify", action="store_true",
                        help="Упростить ONNX-граф при экспорте")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_label_file(label_path: Path, num_classes: int) -> List[int]:
    """
    Возвращает список классов, встреченных в label-файле.
    Проверяет базовую корректность YOLO-разметки.
    """
    found_classes = []

    if not label_path.exists():
        raise FileNotFoundError(f"Не найден label-файл: {label_path}")

    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return found_classes

    for line_idx, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}: строка {line_idx} должна содержать 5 значений, получено {len(parts)}"
            )

        cls_id = int(float(parts[0]))
        if cls_id < 0 or cls_id >= num_classes:
            raise ValueError(
                f"{label_path}: class id {cls_id} вне диапазона [0, {num_classes - 1}]"
            )

        coords = list(map(float, parts[1:]))
        for value in coords:
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"{label_path}: координаты YOLO должны быть в диапазоне [0, 1], найдено {value}"
                )

        found_classes.append(cls_id)

    return found_classes


def collect_pairs(raw_dataset: Path, class_names: List[str]) -> List[Tuple[Path, Path, List[int]]]:
    images_dir = raw_dataset / "images"
    labels_dir = raw_dataset / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Не найдена папка с изображениями: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Не найдена папка с разметкой: {labels_dir}")

    pairs: List[Tuple[Path, Path, List[int]]] = []
    num_classes = len(class_names)

    for image_path in sorted(images_dir.rglob("*")):
        if image_path.suffix.lower() not in IMG_EXTS:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(
                f"Для изображения {image_path.name} не найден label {label_path.name}"
            )

        classes_in_file = read_label_file(label_path, num_classes)
        pairs.append((image_path, label_path, classes_in_file))

    if not pairs:
        raise RuntimeError("В raw_dataset/images не найдено ни одного изображения")

    return pairs


def compute_split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    if total < 3:
        raise ValueError("Для train/val/test нужно минимум 3 изображения")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("Сумма train/val/test ratios должна быть равна 1.0")

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    # гарантируем хотя бы по 1 примеру
    if train_count == 0:
        train_count = 1
    if val_count == 0:
        val_count = 1
    if test_count == 0:
        test_count = 1

    while train_count + val_count + test_count > total:
        if train_count >= val_count and train_count >= test_count and train_count > 1:
            train_count -= 1
        elif val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    while train_count + val_count + test_count < total:
        train_count += 1

    return train_count, val_count, test_count


def copy_split_items(items: List[Tuple[Path, Path, List[int]]], split_name: str, out_root: Path) -> None:
    img_out = out_root / "images" / split_name
    lbl_out = out_root / "labels" / split_name
    ensure_dir(img_out)
    ensure_dir(lbl_out)

    for idx, (img_path, lbl_path, _) in enumerate(items):
        new_stem = f"{idx:05d}_{img_path.stem}"
        new_img = img_out / f"{new_stem}{img_path.suffix.lower()}"
        new_lbl = lbl_out / f"{new_stem}.txt"
        shutil.copy2(img_path, new_img)
        shutil.copy2(lbl_path, new_lbl)


def prepare_dataset(raw_dataset: Path, prepared_dataset: Path, class_names: List[str], seed: int,
                    train_ratio: float, val_ratio: float, test_ratio: float) -> Path:
    pairs = collect_pairs(raw_dataset, class_names)

    rng = random.Random(seed)
    rng.shuffle(pairs)

    train_count, val_count, test_count = compute_split_counts(
        total=len(pairs),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    train_items = pairs[:train_count]
    val_items = pairs[train_count:train_count + val_count]
    test_items = pairs[train_count + val_count:]

    if prepared_dataset.exists():
        shutil.rmtree(prepared_dataset)
    ensure_dir(prepared_dataset)

    copy_split_items(train_items, "train", prepared_dataset)
    copy_split_items(val_items, "val", prepared_dataset)
    copy_split_items(test_items, "test", prepared_dataset)

    data_yaml_path = prepared_dataset / "data.yaml"
    data = {
        "path": str(prepared_dataset.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    class_counter = Counter()
    for _, _, classes_in_file in pairs:
        class_counter.update(classes_in_file)

    summary = {
        "total_images": len(pairs),
        "splits": {
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items),
        },
        "classes": class_names,
        "objects_per_class": {
            class_names[i]: class_counter.get(i, 0) for i in range(len(class_names))
        }
    }

    with open(prepared_dataset / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Подготовка датасета завершена ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"data.yaml: {data_yaml_path}")
    return data_yaml_path


def train_model(args: argparse.Namespace, data_yaml_path: Path, class_names: List[str]) -> None:
    print("\n=== Старт обучения ===")
    print(f"Base model: {args.model}")
    print(f"Data YAML : {data_yaml_path}")
    print(f"Project   : {args.project}")
    print(f"Run name  : {args.name}")

    model = YOLO(args.model)

    train_results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=args.seed,

        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        mosaic=0.5,
        mixup=0.0,
    )

    save_dir = Path(getattr(train_results, "save_dir", Path(args.project) / args.name))
    weights_dir = save_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    if not best_pt.exists():
        raise FileNotFoundError(f"После обучения не найден {best_pt}")

    print("\n=== Обучение завершено ===")
    print(f"best.pt: {best_pt}")
    print(f"last.pt: {last_pt if last_pt.exists() else 'не найден'}")

    best_model = YOLO(str(best_pt))

    print("\n=== Валидация best.pt ===")
    val_results = best_model.val(
        data=str(data_yaml_path),
        imgsz=args.imgsz,
        batch=max(1, min(args.batch, 8)),
        device=args.device,
    )

    metrics = {}
    for attr in ("box", "speed"):
        if hasattr(val_results, attr):
            metrics[attr] = str(getattr(val_results, attr))

    with open(save_dir / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_pt": str(best_pt),
            "class_names": class_names,
            "metrics_repr": metrics
        }, f, ensure_ascii=False, indent=2)

    if args.export_onnx:
        print("\n=== Экспорт ONNX ===")
        exported = best_model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.onnx_opset,
            simplify=args.simplify,
            dynamic=False,
        )
        print(f"ONNX exported to: {exported}")


def main() -> None:
    args = parse_args()
    class_names = [x.strip() for x in args.classes.split(",") if x.strip()]
    if len(class_names) < 2:
        raise ValueError("Нужно минимум 2 класса")

    raw_dataset = Path(args.raw_dataset)
    prepared_dataset = Path(args.prepared_dataset)

    data_yaml_path = prepare_dataset(
        raw_dataset=raw_dataset,
        prepared_dataset=prepared_dataset,
        class_names=class_names,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_model(args, data_yaml_path, class_names)


if __name__ == "__main__":
    main()

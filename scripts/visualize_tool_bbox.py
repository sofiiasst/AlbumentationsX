"""Visualize YOLO labels (txt or json) on an image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2


def _read_yolo_txt(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    bboxes = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO line: {line}")
        class_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        bboxes.append((class_id, x_center, y_center, width, height))
    return bboxes


def _read_yolo_json(json_path: Path) -> List[Tuple[int, float, float, float, float]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "tool_bbox_yolo" in data:
        yolo = data["tool_bbox_yolo"]
        return [(
            int(yolo.get("class_id", 0)),
            float(yolo["x_center_norm"]),
            float(yolo["y_center_norm"]),
            float(yolo["width_norm"]),
            float(yolo["height_norm"]),
        )]

    if isinstance(data, dict) and "tool_bbox_normalized" in data:
        values = data["tool_bbox_normalized"]
        if isinstance(values, list) and len(values) == 4:
            return [(0, float(values[0]), float(values[1]), float(values[2]), float(values[3]))]

    if isinstance(data, list) and len(data) == 4:
        return [(0, float(data[0]), float(data[1]), float(data[2]), float(data[3]))]

    raise ValueError(f"Unsupported JSON label format: {json_path}")


def _read_labels(labels_path: Path) -> List[Tuple[int, float, float, float, float]]:
    # Handle common typo: path like tools/drill.4.txt -> tools/drill/4.txt
    if not labels_path.exists() and "." in labels_path.stem:
        parts = labels_path.stem.split(".")
        if len(parts) >= 2:
            dir_candidate = labels_path.parent / parts[0]
            file_candidate = dir_candidate / f"{parts[1]}{labels_path.suffix}"
            if file_candidate.exists():
                labels_path = file_candidate

    if labels_path.suffix.lower() == ".json":
        return _read_yolo_json(labels_path)

    if labels_path.exists():
        return _read_yolo_txt(labels_path)

    # Fallback: try json with same stem
    json_path = labels_path.with_suffix(".json")
    if json_path.exists():
        return _read_yolo_json(json_path)

    # Fallback: if stem is a number, try bbox{n}.json in same folder
    if labels_path.stem.isdigit():
        candidate = labels_path.parent / f"bbox{labels_path.stem}.json"
        if candidate.exists():
            return _read_yolo_json(candidate)

    raise FileNotFoundError(f"Labels not found: {labels_path}")


def _draw_bboxes(
    img,
    bboxes: List[Tuple[int, float, float, float, float]],
    label_width: int,
    label_height: int,
) -> None:
    img_h, img_w = img.shape[:2]
    scale_x = img_w / label_width if label_width else 1.0
    scale_y = img_h / label_height if label_height else 1.0
    for class_id, x_center_n, y_center_n, width_n, height_n in bboxes:
        x_center = x_center_n * label_width * scale_x
        y_center = y_center_n * label_height * scale_y
        width = width_n * label_width * scale_x
        height = height_n * label_height * scale_y

        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"class {class_id}",
            (x_min, max(10, y_min - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO txt on an image")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("labels", help="Path to YOLO .txt labels")
    parser.add_argument("--label-size", default="512x512", help="Label space size, e.g. 512x512")
    parser.add_argument("output", nargs="?", help="Optional output path")
    args = parser.parse_args()

    image_path = Path(args.image)
    labels_path = Path(args.labels)
    if args.output:
        output_path = Path(args.output)
    else:
        repo_root = Path(__file__).resolve().parent.parent
        output_path = repo_root / "tools/outputs/generated_bbox.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    # labels_path can be .txt or .json; existence is checked in _read_labels

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read: {image_path}")

    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    label_w, label_h = 512, 512
    if "x" in args.label_size.lower():
        parts = args.label_size.lower().split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            label_w, label_h = int(parts[0]), int(parts[1])

    bboxes = _read_labels(labels_path)
    _draw_bboxes(img, bboxes, label_w, label_h)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def _round4(value: float) -> float:
    return round(value, 4)


def _get_image_size(image_path: Path) -> Tuple[int, int]:
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def _resolve_image_path(input_data: Dict[str, Any]) -> Path | None:
    key = str(input_data.get("key", "")).strip()
    if not key:
        return None

    stem = Path(key).stem
    digits = "".join(ch for ch in stem if ch.isdigit())

    repo_root = Path(__file__).resolve().parent.parent
    tools_dir = repo_root / "tools" / "drill"

    candidates = []
    if digits:
        candidates.extend([
            tools_dir / f"{digits}.png",
            tools_dir / f"{digits}.jpg",
        ])

    candidates.extend([
        tools_dir / f"{stem}.png",
        tools_dir / f"{stem}.jpg",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _resolve_output_path(input_data: Dict[str, Any]) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    key = str(input_data.get("key", "")).strip()
    stem = Path(key).stem if key else ""
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return repo_root / "tools" / "drill" / f"bbox{digits}.json"
    return repo_root / "tools" / "outputs" / "generated_bbox.json"


def convert_bbox(
    input_data: Dict[str, Any],
    use_image_size: bool = False,
    image_path: Path | None = None,
) -> Dict[str, Any]:
    boxes = input_data.get("boxes") or []
    if not boxes:
        raise ValueError("Input JSON must contain a non-empty 'boxes' list.")

    box = boxes[0]
    label = box.get("label", "tool")

    image_width = float(input_data["width"])
    image_height = float(input_data["height"])
    if use_image_size:
        if image_path is None:
            image_path = _resolve_image_path(input_data)
        if image_path is not None:
            image_width, image_height = _get_image_size(image_path)
            image_width = float(image_width)
            image_height = float(image_height)

    x = float(box["x"])
    y = float(box["y"])
    width = float(box["width"])
    height = float(box["height"])

    # NOTE: This uses x, y as the bbox center as requested by the expected output.
    x_center_norm = _round4(x / image_width)
    y_center_norm = _round4(y / image_height)
    width_norm = _round4(width / image_width)
    height_norm = _round4(height / image_height)

    key = str(input_data.get("key", "tool"))
    tool_name = Path(key).stem  # Extract name without extension

    output = {
        "image_width": int(image_width) if image_width.is_integer() else image_width,
        "image_height": int(image_height) if image_height.is_integer() else image_height,
        "tool_bbox_yolo": {
            "class_id": 0,
            "x_center_norm": x_center_norm,
            "y_center_norm": y_center_norm,
            "width_norm": width_norm,
            "height_norm": height_norm,
        },
        "tool_name": tool_name,
    }

    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert bbox JSON to YOLO normalized format.")
    parser.add_argument("input", nargs="?", default="tools/input.json", help="Path to input JSON")
    parser.add_argument("output", nargs="?", help="Optional output JSON path")
    parser.add_argument("--image-path", help="Optional image path for actual size")
    parser.add_argument(
        "--use-image-size",
        action="store_true",
        help="Use actual image size for normalization",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    image_path = Path(args.image_path) if args.image_path else None
    output_data = convert_bbox(
        input_data,
        use_image_size=args.use_image_size,
        image_path=image_path,
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _resolve_output_path(input_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()

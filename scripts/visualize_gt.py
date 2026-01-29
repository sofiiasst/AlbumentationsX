import argparse
from pathlib import Path
import cv2

COLORS = [
    (255, 87, 51),
    (46, 204, 113),
    (52, 152, 219),
    (155, 89, 182),
    (241, 196, 15),
    (26, 188, 156),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO ground truth labels.")
    parser.add_argument("--run", type=Path, default=Path("outputs/run_1"),
                        help="Run directory (expects images/ and labels/ subdirectories inside).")
    parser.add_argument("--output", type=Path, default=Path("outputs/visualize_bboxes"),
                        help="Where to save annotated images.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optionally stop after this many images.")
    parser.add_argument("--class-names", type=str, nargs="+", default=["tool"],
                        help="Class names (default: tool).")
    return parser.parse_args()

def ensure_output_dir(base: Path) -> Path:
    output_dir = base
    counter = 1
    while output_dir.exists():
        counter += 1
        output_dir = Path(f"{base}_{counter}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def read_labels(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    with label_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            x1 = max(0, (xc - bw / 2) * img_w)
            y1 = max(0, (yc - bh / 2) * img_h)
            x2 = min(img_w, (xc + bw / 2) * img_w)
            y2 = min(img_h, (yc + bh / 2) * img_h)
            boxes.append((cls, (int(x1), int(y1), int(x2), int(y2))))
    return boxes

def annotate(image, boxes, names):
    for cls, (x1, y1, x2, y2) in boxes:
        color = COLORS[cls % len(COLORS)]
        label = names[cls] if names and cls < len(names) else str(cls)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return image

def main():
    args = parse_args()
    run_dir = args.run
    images_dir = run_dir / "images"
    labels_dir = run_dir / "labels"
    names = args.class_names

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return
    output_dir = ensure_output_dir(args.output)
    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.limit:
        image_files = image_files[: args.limit]
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not read image: {img_path}")
            continue
        h, w = image.shape[:2]
        boxes = read_labels(labels_dir / f"{img_path.stem}.txt", w, h)
        annotated = annotate(image, boxes, names)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)
        status = "(no labels)" if not boxes else ""
        print(f"Saved {out_path} {status}")
    print(f"Done. Visuals in: {output_dir}")

if __name__ == "__main__":
    main()

"""Visualize the original tool image with its bounding box."""
import json
import cv2
from pathlib import Path


def visualize_tool_bbox():
    """Draw bbox on the original tool image."""
    repo_root = Path(__file__).resolve().parent.parent
    tool_path = repo_root / "tools/hammer/2.png"
    bbox_path = repo_root / "tools/hammer/bbox2.json"

    if not tool_path.exists():
        print(f"Tool image not found: {tool_path}")
        return

    if not bbox_path.exists():
        print(f"Bbox file not found: {bbox_path}")
        return

    # Load image
    img = cv2.imread(str(tool_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read: {tool_path}")
        return

    # Convert RGBA to RGB for visualization
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Load bbox
    with open(bbox_path) as f:
        bbox_data = json.load(f)

    # Convert YOLO format to pixel coordinates
    img_w = bbox_data["image_width"]
    img_h = bbox_data["image_height"]
    yolo = bbox_data["tool_bbox_yolo"]

    x_center = yolo["x_center_norm"] * img_w
    y_center = yolo["y_center_norm"] * img_h
    width = yolo["width_norm"] * img_w
    height = yolo["height_norm"] * img_h

    x = int(x_center - width / 2)
    y = int(y_center - height / 2)
    w = int(width)
    h = int(height)

    # Draw rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        img,
        "tool bbox",
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Save visualization
    out_path = repo_root / "tools/1_bbox_visualization.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")
    print(f"Tool image size: {img_w}x{img_h}")
    print(f"YOLO: x_center={yolo['x_center_norm']}, y_center={yolo['y_center_norm']}, width={yolo['width_norm']}, height={yolo['height_norm']}")
    print(f"Pixels: x={x}, y={y}, width={w}, height={h}")


if __name__ == "__main__":
    visualize_tool_bbox()


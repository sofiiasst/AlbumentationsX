import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


def find_images(base_path: Path) -> list[Path]:
    """Return all background images in the directory."""
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    images = []
    for ext in extensions:
        images.extend(sorted(base_path.glob(f"*{ext}")))
    return sorted(set(images))  # Remove duplicates and sort


def load_foreground(path: Path) -> np.ndarray:
    fg = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if fg is None:
        raise FileNotFoundError(f"Foreground not found: {path}")
    if fg.ndim == 2:
        fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGRA)
    if fg.shape[2] == 3:
        alpha = np.full((fg.shape[0], fg.shape[1]), 255, dtype=fg.dtype)
        fg = np.dstack((fg, alpha))
    return fg


def paste_foreground(fg_rgba: np.ndarray, bg_bgr: np.ndarray, tool_bbox: dict, rotate_tool: bool = True) -> tuple:
    """Paste foreground on background and return composited image + transformed bbox."""
    
    # Optional: rotate the tool before pasting
    if rotate_tool:
        fg_rgba = A.Rotate(
            limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=1.0
        )(image=fg_rgba)["image"]
        
        # Compute bbox from actual opaque pixels (threshold alpha > 128 to ignore semi-transparent edges)
        alpha_channel = fg_rgba[:, :, 3]
        _, alpha_mask = cv2.threshold(alpha_channel, 128, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(alpha_mask)
        
        if coords is not None:
            x_min, y_min, bbox_w, bbox_h = cv2.boundingRect(coords)
            x_max = x_min + bbox_w
            y_max = y_min + bbox_h
            
            tool_bbox = {
                "x": float(x_min),
                "y": float(y_min),
                "width": float(bbox_w),
                "height": float(bbox_h),
            }
    
    # Randomly scale foreground relative to background width
    scale = random.uniform(0.25, 0.5)
    new_w = max(1, int(bg_bgr.shape[1] * scale))
    new_h = max(1, int(fg_rgba.shape[0] * new_w / fg_rgba.shape[1]))
    
    # Ensure foreground fits within background
    if new_w >= bg_bgr.shape[1]:
        new_w = int(bg_bgr.shape[1] * 0.8)
        new_h = max(1, int(fg_rgba.shape[0] * new_w / fg_rgba.shape[1]))
    if new_h >= bg_bgr.shape[0]:
        new_h = int(bg_bgr.shape[0] * 0.8)
        new_w = max(1, int(fg_rgba.shape[1] * new_h / fg_rgba.shape[0]))
    
    fg_resized = cv2.resize(fg_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    alpha = fg_resized[:, :, 3].astype(float) / 255.0
    alpha_3 = cv2.merge([alpha, alpha, alpha])
    fg_bgr = fg_resized[:, :, :3].astype(float)

    max_x = bg_bgr.shape[1] - new_w
    max_y = bg_bgr.shape[0] - new_h
    if max_x < 0 or max_y < 0:
        return None, None  # Skip this image instead of raising error
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    roi = bg_bgr[y : y + new_h, x : x + new_w].astype(float)
    blended = roi * (1.0 - alpha_3) + fg_bgr * alpha_3
    bg_bgr[y : y + new_h, x : x + new_w] = blended

    # Transform bounding box
    scale_factor = new_w / fg_rgba.shape[1]
    bbox_x = int(tool_bbox["x"] * scale_factor) + x
    bbox_y = int(tool_bbox["y"] * scale_factor) + y
    bbox_w = int(tool_bbox["width"] * scale_factor)
    bbox_h = int(tool_bbox["height"] * scale_factor)
    
    # Skip if bbox is too small
    if bbox_w < 5 or bbox_h < 5:
        return None, None

    # Get final image dimensions (BEFORE augmentation)
    img_h, img_w = bg_bgr.shape[:2]

    # Normalize to YOLO format (0-1) using final image dimensions
    yolo_x_center = (bbox_x + bbox_w / 2) / img_w
    yolo_y_center = (bbox_y + bbox_h / 2) / img_h
    yolo_w = bbox_w / img_w
    yolo_h = bbox_h / img_h

    transformed_bbox = {
        "x": bbox_x,
        "y": bbox_y,
        "width": bbox_w,
        "height": bbox_h,
        "x_min": bbox_x,
        "y_min": bbox_y,
        "x_max": bbox_x + bbox_w,
        "y_max": bbox_y + bbox_h,
        "img_h": img_h,
        "img_w": img_w,
        "yolo_format": f"0 {yolo_x_center:.6f} {yolo_y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}",
    }

    return bg_bgr.astype(np.uint8), transformed_bbox


def main():
    repo_root = Path(__file__).resolve().parent.parent
    hammer_dir = repo_root / "tools/hammer"
    
    # Find all image and bbox pairs
    image_files = sorted([f for f in hammer_dir.glob("*.png") if f.stem.isdigit()])
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {hammer_dir}")
    
    print(f"Found {len(image_files)} tool image(s) to process")
    
    # Load all backgrounds
    bg_dir = repo_root / "data/backgrounds"
    bg_paths = find_images(bg_dir)
    
    if not bg_paths:
        raise FileNotFoundError(f"No background images found in {bg_dir}")
    
    print(f"Found {len(bg_paths)} background image(s)")
    
    backgrounds = {}
    for bg_path in bg_paths:
        bg_bgr = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if bg_bgr is None:
            print(f"Warning: Could not read background {bg_path.name}, skipping...")
            continue
        backgrounds[bg_path.stem] = bg_bgr

    transform = A.Compose(
        [
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3),
    )

    # Find next available run number
    outputs_base = repo_root / "outputs"
    outputs_base.mkdir(parents=True, exist_ok=True)
    
    run_num = 1
    while (outputs_base / f"run_{run_num}").exists():
        run_num += 1
    
    outputs_dir = outputs_base / f"run_{run_num}"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images and labels subdirectories
    images_dir = outputs_dir / "images"
    labels_dir = outputs_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Save annotations list
    all_annotations = []
    
    # Process each image
    for img_idx, fg_path in enumerate(image_files, 1):
        # Find corresponding bbox file
        stem = fg_path.stem  # e.g., "1", "2", etc.
        bbox_path = hammer_dir / f"bbox{stem if stem != '1' else ''}.json"
        
        if not bbox_path.exists():
            print(f"Warning: Bbox file not found for {fg_path.name}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing image {img_idx}/{len(image_files)}: {fg_path.name}")
        print(f"{'='*60}")
        
        # Load original tool bbox
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
        
        tool_bbox = {
            "x": x_center - width / 2,
            "y": y_center - height / 2,
            "width": width,
            "height": height,
        }
        
        fg_rgba = load_foreground(fg_path)
        
        # Generate images for each background
        for bg_stem, bg_bgr in backgrounds.items():
            num_images = 20
            for i in range(num_images):
                bg_bgr_copy = bg_bgr.copy()
                composed, bbox = paste_foreground(fg_rgba, bg_bgr_copy, tool_bbox)
                
                if composed is None:
                    continue

                # Prepare bbox for Albumentations (pascal_voc)
                bboxes = [[bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]]]
                class_labels = ["tool"]

                transformed = transform(image=composed, bboxes=bboxes, class_labels=class_labels)
                augmented = transformed["image"]
                out_bboxes = transformed["bboxes"]

                if out_bboxes is None or len(out_bboxes) == 0:
                    print(f"    [{i+1}/{num_images}] Skipped (bbox lost after transform)")
                    continue

                x_min, y_min, x_max, y_max = out_bboxes[0]
                aug_h, aug_w = augmented.shape[:2]

                bbox_w = x_max - x_min
                bbox_h = y_max - y_min
                bbox_x = x_min
                bbox_y = y_min

                # Normalize to YOLO format
                yolo_x_center = (bbox_x + bbox_w / 2) / aug_w
                yolo_y_center = (bbox_y + bbox_h / 2) / aug_h
                yolo_w = bbox_w / aug_w
                yolo_h = bbox_h / aug_h

                yolo_str = f"0 {yolo_x_center:.6f} {yolo_y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}"

                filename = f"tool_{stem}_on_{bg_stem}_{i:03d}.jpg"
                out_path = images_dir / filename
                cv2.imwrite(str(out_path), augmented)
                print(f"    [{i+1}/{num_images}] Saved {filename}")

                # Save YOLO format txt file
                txt_filename = f"tool_{stem}_on_{bg_stem}_{i:03d}.txt"
                txt_path = labels_dir / txt_filename
                with open(txt_path, "w") as f:
                    f.write(yolo_str)

                # Store annotation
                all_annotations.append({
                    "image": filename,
                    "source_image": fg_path.name,
                    "background": bg_stem,
                    "bbox": {
                        "x": bbox_x,
                        "y": bbox_y,
                        "width": bbox_w,
                        "height": bbox_h,
                        "x_min": bbox_x,
                        "y_min": bbox_y,
                        "x_max": x_max,
                        "y_max": y_max,
                        "img_h": aug_h,
                        "img_w": aug_w,
                        "yolo_format": yolo_str,
                    },
                })

    # Save all annotations to JSON
    annotations_path = outputs_dir / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(all_annotations, f, indent=2)
    print(f"\n{'='*60}")
    print(f"All annotations saved to {annotations_path}")
    print(f"Total images generated: {len(all_annotations)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

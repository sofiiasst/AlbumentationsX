import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


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


def rotate_image_with_expansion(image, angle):
    """Rotate image and expand canvas to fit the rotated content"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate the rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate the bounding box of the rotated image
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to account for the new center
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    
    # Rotate with expanded canvas
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    return rotated


def paste_single_tool(
    fg_rgba: np.ndarray,
    bg_bgr: np.ndarray,
    x_pos: int,
    y_pos: int
) -> np.ndarray:
    """Paste a single tool at specified position."""
    fg_h, fg_w = fg_rgba.shape[:2]
    
    # Ensure it fits
    if x_pos + fg_w > bg_bgr.shape[1] or y_pos + fg_h > bg_bgr.shape[0]:
        raise ValueError("Tool doesn't fit at specified position")
    
    alpha = fg_rgba[:, :, 3].astype(float) / 255.0
    alpha_3 = cv2.merge([alpha, alpha, alpha])
    fg_bgr = fg_rgba[:, :, :3].astype(float)
    
    roi = bg_bgr[y_pos:y_pos + fg_h, x_pos:x_pos + fg_w].astype(float)
    blended = roi * (1.0 - alpha_3) + fg_bgr * alpha_3
    bg_bgr[y_pos:y_pos + fg_h, x_pos:x_pos + fg_w] = blended
    
    return bg_bgr


def main():
    repo_root = Path(__file__).resolve().parent.parent
    
    # Load drill/3.png and its bbox
    drill_path = repo_root / "tools_source/drill/3.png"
    bbox_path = repo_root / "tools_source/drill/bbox3.json"
    bg_path = repo_root / "data/backgrounds/background_1.jpg"
    
    print(f"Loading foreground: {drill_path}")
    fg_rgba_original = load_foreground(drill_path)
    
    print(f"Loading background: {bg_path}")
    bg_bgr = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
    if bg_bgr is None:
        raise FileNotFoundError(f"Background not found: {bg_path}")
    
    # Load bbox data
    with open(bbox_path) as f:
        bbox_data = json.load(f)
    
    print("\n" + "="*60)
    print("STEP 1: Creating base image with two drills")
    print("="*60)
    
    # Create base image with two drills
    base_image = bg_bgr.copy()
    
    # First drill: scaled smaller (scale factor 0.25)
    scale_factor_1 = 0.25
    new_w_1 = int(bg_bgr.shape[1] * scale_factor_1)
    new_h_1 = int(fg_rgba_original.shape[0] * new_w_1 / fg_rgba_original.shape[1])
    fg_small = cv2.resize(fg_rgba_original, (new_w_1, new_h_1), interpolation=cv2.INTER_AREA)
    
    # Position first drill on the left (higher up - background, but brought closer)
    x1 = int(bg_bgr.shape[1] * 0.15)
    y1 = int(bg_bgr.shape[0] * 0.35)  # Moved up a bit
    print(f"  - Pasting first drill (scaled small: {scale_factor_1}) at ({x1}, {y1})")
    base_image = paste_single_tool(fg_small, base_image, x1, y1)
    
    # Second drill: rotated 20 degrees with canvas expansion
    print(f"  - Rotating second drill by 20 degrees...")
    fg_rotated = rotate_image_with_expansion(fg_rgba_original, 20)
    
    # Crop rotated drill to remove black padding
    alpha_channel = fg_rotated[:, :, 3]
    _, alpha_mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(alpha_mask)
    if coords is not None:
        x_r, y_r, w_r, h_r = cv2.boundingRect(coords)
        fg_rotated = fg_rotated[y_r:y_r+h_r, x_r:x_r+w_r]
    
    # Don't flip yet - will be flipped in augmentations
    
    # Scale rotated drill based on HEIGHT to ensure vertical fit
    bg_h, bg_w = bg_bgr.shape[:2]
    print(f"  - Background size: {bg_w}x{bg_h}")
    
    max_drill_height = int(bg_h * 0.40)  # Max 40% of background height (reduced from 60%)
    scale_ratio = max_drill_height / fg_rotated.shape[0]
    new_h_2 = max_drill_height
    new_w_2 = int(fg_rotated.shape[1] * scale_ratio)
    fg_rotated_scaled = cv2.resize(fg_rotated, (new_w_2, new_h_2), interpolation=cv2.INTER_AREA)
    
    # Position second drill - lower on screen (closer to viewer/foreground)
    x2 = int(bg_w * 0.55)
    y2 = int(bg_h * 0.55)  # Moved up a bit (was 0.60)
    
    # Adjust if drill would exceed bounds
    if x2 + new_w_2 > bg_w:
        x2 = bg_w - new_w_2 - 10
    if y2 + new_h_2 > bg_h:
        y2 = bg_h - new_h_2 - 10
    
    print(f"  - Final rotated drill size: {new_w_2}x{new_h_2}, pos: ({x2}, {y2})")
    
    print(f"  - Pasting second drill (rotated) at ({x2}, {y2})")
    base_image = paste_single_tool(fg_rotated_scaled, base_image, x2, y2)
    
    # Save base image
    output_dir = repo_root / "outputs/augmentation_stages"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_output = output_dir / "0_base_two_drills.jpg"
    cv2.imwrite(str(base_output), base_image)
    print(f"  ✓ Saved: {base_output}")
    
    # Define individual augmentations
    augmentations = [
        ("1_horizontal_flip", A.HorizontalFlip(p=1.0)),
        ("2_brightness_contrast", A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0)),
        ("3_shadow", A.RandomShadow(shadow_roi=(0.2, 0.4, 0.7, 0.8), num_shadows_limit=(1, 1), shadow_dimension=4, p=1.0)),
        ("4_blur", A.GaussianBlur(blur_limit=(7, 13), p=1.0)),
    ]
    
    print("\n" + "="*60)
    print("STEP 2: Applying each augmentation separately")
    print("="*60)
    
    # Apply each augmentation separately
    for name, transform in augmentations:
        aug_compose = A.Compose([transform])
        augmented = aug_compose(image=base_image)["image"]
        
        output_path = output_dir / f"{name}.jpg"
        cv2.imwrite(str(output_path), augmented)
        print(f"  ✓ Saved: {output_path}")
    
    print("\n" + "="*60)
    print("STEP 3: Applying all augmentations combined")
    print("="*60)
    
    # Apply all augmentations combined
    combined_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.RandomShadow(shadow_roi=(0.2, 0.4, 0.7, 0.8), num_shadows_limit=(1, 1), shadow_dimension=4, p=0.3),
        A.GaussianBlur(blur_limit=(7, 13), p=0.35),
    ])
    
    combined_augmented = combined_transform(image=base_image)["image"]
    combined_output = output_dir / "5_all_combined.jpg"
    cv2.imwrite(str(combined_output), combined_augmented)
    print(f"  ✓ Saved: {combined_output}")
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

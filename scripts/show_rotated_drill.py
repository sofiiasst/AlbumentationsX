import cv2
from pathlib import Path
import numpy as np
import math


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


def main():
    repo_root = Path(__file__).resolve().parent.parent
    
    # Load drill
    drill_path = repo_root / "tools_source/drill/3.png"
    
    print(f"Loading drill: {drill_path}")
    fg_rgba = load_foreground(drill_path)
    
    print(f"Original drill shape: {fg_rgba.shape}")
    
    # Rotate with proper canvas expansion
    print("Rotating 20 degrees with canvas expansion...")
    fg_rotated = rotate_image_with_expansion(fg_rgba, 20)
    
    print(f"After rotation shape: {fg_rotated.shape}")
    
    # Save the rotated drill
    output_dir = repo_root / "outputs/augmentation_stages"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "rotated_drill_20deg_expanded.png"
    cv2.imwrite(str(output_path), fg_rotated)
    print(f"✓ Saved rotated drill (expanded canvas): {output_path}")
    
    # Also show the cropped version (what we use in the script)
    print("\nCropping to remove black padding...")
    alpha_channel = fg_rotated[:, :, 3]
    _, alpha_mask = cv2.threshold(alpha_channel, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(alpha_mask)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        fg_cropped = fg_rotated[y:y+h, x:x+w]
        print(f"Cropped drill shape: {fg_cropped.shape}")
        
        cropped_path = output_dir / "rotated_drill_20deg_cropped.png"
        cv2.imwrite(str(cropped_path), fg_cropped)
        print(f"✓ Saved cropped rotated drill: {cropped_path}")


if __name__ == "__main__":
    main()

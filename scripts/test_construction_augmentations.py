"""Test recommended augmentations for construction site tool detection."""
import argparse
from pathlib import Path

import albumentations as A
import cv2


def test_construction_augmentations(image_path: Path, output_dir: Path):
    """Apply recommended augmentations for construction site tools."""
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Define augmentations recommended for construction site tool detection
    augmentations = {
        "01_original": None,
        
        # Lighting variations
        "02_brightness_contrast": A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        "03_hue_saturation": A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=1.0),
        "04_random_shadow": A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2), p=1.0),
        
        # Motion/Blur
        "05_gaussian_blur": A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        "06_motion_blur": A.MotionBlur(blur_limit=5, p=1.0),
        
        # Noise
        "07_gauss_noise": A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        
        # Geometric
        "08_horizontal_flip": A.HorizontalFlip(p=1.0),
        "09_affine_rotate": A.Affine(rotate=(-25, 25), p=1.0),
        "10_perspective": A.Perspective(scale=(0.05, 0.1), p=1.0),
        
        # Combined realistic augmentation
        "11_combined_example": A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.Affine(rotate=(-25, 25), scale=(0.8, 1.2), p=1.0),
        ]),
    }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply and save each augmentation
    print(f"\n{'='*70}")
    print(f"Testing CONSTRUCTION SITE augmentations on: {image_path.name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    for name, transform in augmentations.items():
        if transform is None:
            # Original image
            result = img.copy()
        else:
            # Apply augmentation
            result = transform(image=img)["image"]
        
        # Save result
        output_path = output_dir / f"{name}.jpg"
        cv2.imwrite(str(output_path), result)
        
        # Format output
        display_name = name.split('_', 1)[1] if '_' in name else name
        print(f"✓ {display_name:30s} -> {output_path.name}")
    
    print(f"\n{'='*70}")
    print(f"✅ All {len(augmentations)} augmentations saved!")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Test construction site augmentations")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output-dir", help="Output directory", default=None)
    args = parser.parse_args()
    
    image_path = Path(args.image)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: outputs/construction_augmentations/<image_stem>
        repo_root = Path(__file__).resolve().parent.parent
        output_dir = repo_root / "outputs" / "construction_augmentations" / image_path.stem
    
    test_construction_augmentations(image_path, output_dir)


if __name__ == "__main__":
    main()

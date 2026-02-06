"""Test different augmentation techniques on a single image."""
import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


def test_augmentations(image_path: Path, output_dir: Path):
    """Apply various augmentations to an image and save results."""
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Define augmentations to test
    augmentations = {
        "original": None,
        "gaussian_blur": A.GaussianBlur(blur_limit=(7, 7), p=1.0),
        "motion_blur": A.MotionBlur(blur_limit=15, p=1.0),
        "median_blur": A.MedianBlur(blur_limit=7, p=1.0),
        "defocus": A.Defocus(radius=(3, 5), p=1.0),
        "brightness_contrast": A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        "hue_saturation": A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        "color_jitter": A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        "rgb_shift": A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        "random_gamma": A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        "clahe": A.CLAHE(clip_limit=4.0, p=1.0),
        "sharpen": A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        "emboss": A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
        "horizontal_flip": A.HorizontalFlip(p=1.0),
        "vertical_flip": A.VerticalFlip(p=1.0),
        "rotate_15": A.Rotate(limit=15, p=1.0),
        "rotate_45": A.Rotate(limit=45, p=1.0),
        "shift_scale_rotate": A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=1.0),
        "perspective": A.Perspective(scale=(0.05, 0.1), p=1.0),
        "affine": A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=1.0),
        "elastic": A.ElasticTransform(alpha=1, sigma=50, p=1.0),
        "grid_distortion": A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        "optical_distortion": A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
        "gauss_noise": A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        "iso_noise": A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        "random_shadow": A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2), p=1.0),
        "random_rain": A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, p=1.0),
        "random_fog": A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, p=1.0),
        "random_snow": A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=1.0),
        "random_sun_flare": A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, p=1.0),
        "solarize": A.Solarize(threshold=128, p=1.0),
        "posterize": A.Posterize(num_bits=4, p=1.0),
        "equalize": A.Equalize(p=1.0),
        "invert": A.InvertImg(p=1.0),
        "channel_shuffle": A.ChannelShuffle(p=1.0),
        "to_gray": A.ToGray(p=1.0),
        "image_compression": A.ImageCompression(quality_lower=10, quality_upper=20, p=1.0),
        "downscale": A.Downscale(scale_min=0.25, scale_max=0.5, p=1.0),
        "coarse_dropout": A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=1.0),
        "grid_dropout": A.GridDropout(ratio=0.5, p=1.0),
    }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply and save each augmentation
    print(f"Testing {len(augmentations)} augmentations on {image_path.name}")
    print(f"Output directory: {output_dir}\n")
    
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
        print(f"✓ {name:25s} -> {output_path.name}")
    
    print(f"\n{'='*60}")
    print(f"All {len(augmentations)} augmentations saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Test augmentations on an image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output-dir", help="Output directory", default=None)
    args = parser.parse_args()
    
    image_path = Path(args.image)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: outputs/test_augmentations/<image_stem>
        repo_root = Path(__file__).resolve().parent.parent
        output_dir = repo_root / "outputs" / "test_augmentations" / image_path.stem
    
    test_augmentations(image_path, output_dir)


if __name__ == "__main__":
    main()

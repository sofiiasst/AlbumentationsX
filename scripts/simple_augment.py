import albumentations as A
import cv2
from pathlib import Path


def main():
    # Define augmentation pipeline
    transform = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

    # Input image path
    image_path = Path("image.jpg")
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path.resolve()}")

    # Convert BGR to RGB for Albumentations
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply augmentations
    result = transform(image=image)
    transformed_image = result["image"]

    # Convert back to BGR for OpenCV write
    output_path = Path("augmented.jpg")
    cv2.imwrite(str(output_path), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

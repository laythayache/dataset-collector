from albumentations import (
    Compose, RandomBrightnessContrast, HueSaturationValue,
    MotionBlur, ShiftScaleRotate
)

# This pipeline applies random transforms to diversify your dataset.
augmentation = Compose([
    RandomBrightnessContrast(p=0.5),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
    MotionBlur(blur_limit=7, p=0.3),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
], p=1.0)

if __name__ == '__main__':
    print("Data augmentation pipeline created.")

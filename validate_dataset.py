def validate_dataset(dataset_path):
    """
    Walks through the dataset folder and validates each JPEG image.
    Checks that each image is 1920x1080 and that a corresponding landmarks file exists.
    """
    from PIL import Image
    import os
    errors = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg'):
                try:
                    img = Image.open(os.path.join(root, file))
                    if img.size != (1920, 1080):
                        errors.append(f"Size mismatch: {file}")
                except Exception as e:
                    errors.append(f"Corrupt file: {file} - {str(e)}")
                landmark_file = os.path.join(root, file.replace('.jpg', '_landmarks.npy'))
                if not os.path.exists(landmark_file):
                    errors.append(f"Missing landmarks: {file}")
    return errors

if __name__ == '__main__':
    import os
    ds_path = 'dataset'
    errs = validate_dataset(ds_path)
    if errs:
        print("Validation errors found:")
        for err in errs:
            print(err)
    else:
        print("Dataset is valid.")

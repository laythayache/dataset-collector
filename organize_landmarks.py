import os
import shutil

def organize_landmarks(dataset_root):
    """
    Organizes all .npy landmark files by letters.
    
    The new structure will be:
      dataset_root/
        ├── organized_landmarks/
        │     ├── أ/
        │     │     ├── person1_session1_timestamp_landmarks.npy
        │     │     ├── person2_session2_timestamp_landmarks.npy
        │     │     └── ...
        │     ├── ب/
        │     │     ├── person1_session1_timestamp_landmarks.npy
        │     │     ├── person2_session2_timestamp_landmarks.npy
        │     │     └── ...
        │     └── ...
    """
    raw_data_path = os.path.join(dataset_root, "raw")
    organized_path = os.path.join(dataset_root, "organized_landmarks")
    
    if not os.path.exists(raw_data_path):
        print(f"Raw data path '{raw_data_path}' does not exist.")
        return
    
    os.makedirs(organized_path, exist_ok=True)
    
    for root, _, files in os.walk(raw_data_path):
        for file in files:
            if file.endswith('_landmarks.npy'):
                letter = root.split(os.sep)[-3]
                letter_folder = os.path.join(organized_path, letter)
                os.makedirs(letter_folder, exist_ok=True)
                
                src_file = os.path.join(root, file)
                dst_file = os.path.join(letter_folder, file)
                
                shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} to {dst_file}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(script_dir, "dataset")
    organize_landmarks(dataset_root)

import os
import shutil

def organize_landmarks(dataset_root):
    """
    Organizes .npy landmark files by Arabic letters.
    
    New Structure:
      dataset/
      ├── organized_landmarks/
      │     ├── أ/
      │     │     ├── letter_person_session_timestamp_0001_landmarks.npy
      │     │     ├── letter_person_session_timestamp_0002_landmarks.npy
      │     ├── ب/
      │     ├── ...
    """
    raw_data_path = os.path.join(dataset_root, "raw")
    organized_path = os.path.join(dataset_root, "organized_landmarks")
    
    if not os.path.exists(raw_data_path):
        print(f"Raw data path '{raw_data_path}' does not exist.")
        return
    
    os.makedirs(organized_path, exist_ok=True)
    
    # Walk through the raw data directory
    for person in os.listdir(raw_data_path):
        person_path = os.path.join(raw_data_path, person)
        if not os.path.isdir(person_path):
            continue
        
        for letter in os.listdir(person_path):
            letter_path = os.path.join(person_path, letter)
            if not os.path.isdir(letter_path):
                continue
            
            for session in os.listdir(letter_path):
                session_path = os.path.join(letter_path, session, "landmarks")
                if not os.path.exists(session_path):
                    continue
                
                # Create destination folder by letter
                dest_folder = os.path.join(organized_path, letter)
                os.makedirs(dest_folder, exist_ok=True)
                
                # Copy landmarks to the organized folder
                for file in os.listdir(session_path):
                    if file.endswith("_landmarks.npy"):
                        src_file = os.path.join(session_path, file)
                        dst_file = os.path.join(dest_folder, file)
                        
                        shutil.copy2(src_file, dst_file)
                        print(f"Copied {src_file} -> {dst_file}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(script_dir, "dataset")
    organize_landmarks(dataset_root)
    print("\n✅ Landmarks organized successfully!")

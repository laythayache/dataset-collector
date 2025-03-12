import cv2
import mediapipe as mp
import csv
import time
import os
import numpy as np
from datetime import datetime

# Define a fixed dataset root path using forward slashes
DATASET_ROOT = r"C:/Users/Layth/Desktop/New folder/dataset"

class DataCollector:
    """
    Captures images from your camera along with hand landmarks.
    Data is stored under:
      dataset/raw/{person_name}/{letter}/{session_name_timestamp}/
    Two subfolders are created:
      - images/    -> for JPEG photos
      - landmarks/ -> for landmark .npy files
    Filenames include the letter, person name, session name, timestamp, and an index.
    """

    def __init__(self, person_name="Unknown", session_name="Session"):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.person_name = person_name
        self.session_name = session_name
        # Additional metadata if needed
        self.metadata = {
            'handedness': 'right',
            'skin_tone': 'type_IV',
            'environment': 'indoor_daylight'
        }

    def capture_session(self, letter: str, num_samples: int, session_name: str = None):
        # If session_name isn't provided, use self.session_name
        if not session_name:
            session_name = self.session_name
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        if not cap.isOpened():
            print("[ERROR] Could not open camera.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_folder = f"dataset/raw/{{}}/{{}}/{{}}_{{}}".format(
            self.metadata['person_name'], letter, session_name, timestamp)
        # Create two subfolders: one for images, one for landmarks
        images_folder = os.path.join(base_folder, "images")
        landmarks_folder = os.path.join(base_folder, "landmarks")
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(landmarks_folder, exist_ok=True)
        # Images will be stored in the short_images folder
        images_folder = os.path.join(DATASET_ROOT, "short_images")
        os.makedirs(images_folder, exist_ok=True)
        
        metadata_entries = []
        
        for i in range(num_samples):
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] No frame from camera.")
                continue
                
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            filename = f"{{}}_{{}}_{{}}_{{:04d}}".format(letter, self.metadata['person_name'], session_name + "_" + timestamp, i)
            image_path = os.path.join(images_folder, f"{{}}.jpg".format(filename))
            cv2.imwrite(image_path, frame)
            
            landmark_path = ""
            if results.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
                landmark_path = os.path.join(landmarks_folder, f"{{}}_landmarks.npy".format(filename))
                np.save(landmark_path, landmarks)
                print("[DEBUG] Saved landmarks ->", landmark_path)
            
            metadata_entries.append({
                'filename': filename,
                'capture_time': time.time(),
                'landmarks_detected': bool(results.multi_hand_landmarks),
                'num_hands': len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            })
        
        self._update_metadata_csv(base_folder, metadata_entry)
        cap.release()
        self._update_metadata_csv(base_folder, metadata_entries)
    
    def _update_metadata_csv(self, base_folder, entries):
        csv_path = os.path.join(base_folder, "metadata.csv")
        if not entries:
            print("[INFO] No images captured, skipping CSV.")
            return
        fieldnames = list(entries[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)
        print("[DEBUG] Wrote metadata CSV ->", csv_path)

if __name__ == '__main__':
    collector = DataCollector("Alice", "MorningSession")
    collector.capture_session('أ', 5)

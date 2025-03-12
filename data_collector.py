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
      DATASET_ROOT/raw/{person_name}/{letter}/{session_name}_{timestamp}/landmarks/
    Images are saved in:
      DATASET_ROOT/short_images/
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
        self.metadata = {
            'person_name': person_name,
            'handedness': 'right',
            'skin_tone': 'type_IV',
            'environment': 'indoor_daylight',
            'camera_settings': {
                'resolution': (1920, 1080),
                'exposure': 'auto',
                'white_balance': 5500
            }
        }

    def capture_session(self, letter: str, num_samples: int, session_name: str):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Landmarks will be stored in the long folder structure
        session_folder = os.path.join(DATASET_ROOT, "raw", self.metadata['person_name'], letter, f"{session_name}_{timestamp}")
        landmarks_folder = os.path.join(session_folder, "landmarks")
        os.makedirs(landmarks_folder, exist_ok=True)
        # Images will be stored in the short_images folder
        images_folder = os.path.join(DATASET_ROOT, "short_images")
        os.makedirs(images_folder, exist_ok=True)
        
        metadata_entry = []
        
        for i in range(num_samples):
            ret, frame = cap.read()
            if not ret:
                continue
                
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            filename = f"{letter}_{self.metadata['person_name']}_{session_name}_{timestamp}_{i:04d}"
            image_path = os.path.join(images_folder, f"{filename}.jpg")
            cv2.imwrite(image_path, frame)
            
            landmark_path = ""
            if results.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
                landmark_path = os.path.join(landmarks_folder, f"{filename}_landmarks.npy")
                np.save(landmark_path, landmarks)
            
            metadata_entry.append({
                'filename': filename,
                'capture_time': time.time(),
                'landmarks_detected': bool(results.multi_hand_landmarks),
                'num_hands': len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            })
        
        self._update_metadata_csv(session_folder, metadata_entry)
        cap.release()

    def _update_metadata_csv(self, base_folder, entries):
        csv_path = os.path.join(base_folder, "metadata.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=entries[0].keys())
            writer.writeheader()
            writer.writerows(entries)

if __name__ == '__main__':
    collector = DataCollector("Alice", "MorningSession")
    collector.capture_session('أ', 5, "MorningSession")

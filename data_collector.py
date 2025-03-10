import cv2
import mediapipe as mp
import csv
import time
import os
import numpy as np
from datetime import datetime

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
        base_folder = f"dataset/raw/{{}}/{{}}/{{}}_{{}}".format(
            self.metadata['person_name'], letter, session_name, timestamp)
        # Create two subfolders: one for images, one for landmarks
        images_folder = os.path.join(base_folder, "images")
        landmarks_folder = os.path.join(base_folder, "landmarks")
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(landmarks_folder, exist_ok=True)
        
        metadata_entry = []
        
        for i in range(num_samples):
            ret, frame = cap.read()
            if not ret:
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
            
            metadata_entry.append({
                'filename': filename,
                'capture_time': time.time(),
                'landmarks_detected': bool(results.multi_hand_landmarks),
                'num_hands': len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            })
        
        self._update_metadata_csv(base_folder, metadata_entry)
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

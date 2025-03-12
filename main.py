import os
import csv
import cv2
import mediapipe as mp
import time
import numpy as np
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
from albumentations import Compose, RandomBrightnessContrast, HueSaturationValue, MotionBlur, ShiftScaleRotate

# Define a fixed dataset root using forward slashes
DATASET_ROOT = r"C:/Users/Layth/Desktop/New folder/dataset"

###############################################
# FUNCTIONS TO CREATE PROJECT STRUCTURE & FILES
###############################################

def create_project_structure():
    """
    Creates the project directory structure and code files.
    
    The folder structure created is:
    
      DATASET_ROOT/  (i.e., C:/Users/Layth/Desktop/New folder/dataset)
        ├── metadata.csv         -> Logs details for each captured image.
        ├── raw/                 -> Contains raw captured data (organized as: person/letter/session/).
        ├── processed/           -> For cleaned and split data (train, val, test).
        │      ├── train/
        │      ├── val/
        │      └── test/
        ├── docs/                -> Documentation (e.g., consent forms, guides).
        │       └── ethical_consent_forms/
        └── short_images/        -> Contains captured images (using a short relative path).
    """
    try:
        dataset_root = DATASET_ROOT
        
        # Create main dataset folder and subdirectories
        os.makedirs(dataset_root, exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "raw"), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "processed", "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "processed", "val"), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "processed", "test"), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "docs", "ethical_consent_forms"), exist_ok=True)
        # Create a separate folder for images (short path)
        os.makedirs(os.path.join(dataset_root, "short_images"), exist_ok=True)
        
        # Create a default metadata.csv if it does not exist
        metadata_csv = os.path.join(dataset_root, "metadata.csv")
        if not os.path.exists(metadata_csv):
            with open(metadata_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "filename", "person_name", "session_name", "letter",
                    "capture_time", "light_condition", "skin_tone",
                    "handedness", "landmark_confidence"
                ])
        
        # Create code files (basic content provided)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        create_data_collector_file(script_dir)
        create_validate_dataset_file(script_dir)
        create_augment_dataset_file(script_dir)
        
        messagebox.showinfo("Success", "Project structure and files created successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create project structure:\n{str(e)}")

def create_data_collector_file(script_dir):
    content = f'''import cv2
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
      DATASET_ROOT/raw/{{person_name}}/{{letter}}/{{session_name}}_{{timestamp}}/landmarks/
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
        self.metadata = {{
            'person_name': person_name,
            'handedness': 'right',
            'skin_tone': 'type_IV',
            'environment': 'indoor_daylight',
            'camera_settings': {{
                'resolution': (1920, 1080),
                'exposure': 'auto',
                'white_balance': 5500
            }}
        }}

    def capture_session(self, letter: str, num_samples: int, session_name: str):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Landmarks will be stored in the long folder structure
        session_folder = os.path.join(DATASET_ROOT, "raw", self.metadata['person_name'], letter, f"{{session_name}}_{{timestamp}}")
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
            
            filename = f"{{letter}}_{{self.metadata['person_name']}}_{{session_name}}_{{timestamp}}_{{i:04d}}"
            image_path = os.path.join(images_folder, f"{{filename}}.jpg")
            cv2.imwrite(image_path, frame)
            
            landmark_path = ""
            if results.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
                landmark_path = os.path.join(landmarks_folder, f"{{filename}}_landmarks.npy")
                np.save(landmark_path, landmarks)
            
            metadata_entry.append({{
                'filename': filename,
                'capture_time': time.time(),
                'landmarks_detected': bool(results.multi_hand_landmarks),
                'num_hands': len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            }})
        
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
'''
    file_path = os.path.join(script_dir, "data_collector.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def create_validate_dataset_file(script_dir):
    content = f'''def validate_dataset(dataset_path):
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
                        errors.append(f"Size mismatch: {{file}}")
                except Exception as e:
                    errors.append(f"Corrupt file: {{file}} - {{str(e)}}")
                landmark_file = os.path.join(root, file.replace('.jpg', '_landmarks.npy'))
                if not os.path.exists(landmark_file):
                    errors.append(f"Missing landmarks: {{file}}")
    return errors

if __name__ == '__main__':
    import os
    DATASET_ROOT = r"C:/Users/Layth/Desktop/New folder/dataset"
    ds_path = DATASET_ROOT
    errs = validate_dataset(ds_path)
    if errs:
        print("Validation errors found:")
        for err in errs:
            print(err)
    else:
        print("Dataset is valid.")
'''
    file_path = os.path.join(script_dir, "validate_dataset.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def create_augment_dataset_file(script_dir):
    content = '''from albumentations import (
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
'''
    file_path = os.path.join(script_dir, "augment_dataset.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

###############################################
# DATA AUGMENTATION FUNCTION & PIPELINE
###############################################

augmentation = Compose([
    RandomBrightnessContrast(p=0.5),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
    MotionBlur(blur_limit=7, p=0.3),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
], p=1.0)

def apply_augmentation(image):
    augmented = augmentation(image=image)
    print("Augmentation applied")  # Debug statement to verify augmentation
    return augmented['image']

###############################################
# MAIN GUI: Dataset Creation Navigator
###############################################

class DatasetCreationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Creation Navigator")
        self.root.geometry("900x750")
        
        # Initialize MediaPipe for live preview in Data Collection tab
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera capture for data collection preview
        self.cap_collection = cv2.VideoCapture(0)
        if not self.cap_collection.isOpened():
            messagebox.showerror("Error", "Could not access the camera for preview!")
            self.root.destroy()
        
        # Create Notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')
        
        # Tab 1: Project Setup
        self.setup_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.setup_frame, text="Project Setup")
        self.create_setup_tab()
        
        # Tab 2: Data Collection
        self.collection_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.collection_frame, text="Data Collection")
        self.create_collection_tab()
        
        # Tab 3: Validation
        self.validation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.validation_frame, text="Validation")
        self.create_validation_tab()
        
        # Tab 4: Augmentation Info
        self.augmentation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.augmentation_frame, text="Augmentation")
        self.create_augmentation_tab()
        
        # Tab 5: Instructions & Manual
        self.instructions_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.instructions_frame, text="Instructions")
        self.create_instructions_tab()
    
    def create_setup_tab(self):
        lbl = ttk.Label(self.setup_frame, text="Project Setup", font=("Arial", 18, "bold"))
        lbl.pack(pady=10)
        
        instructions = (
            "This section will create the necessary project structure and code files for your sign language dataset. \n\n"
            "Folder Structure:\n"
            " - dataset/               -> Main folder for your dataset.\n"
            "   - metadata.csv         -> Logs details for each image.\n"
            "   - raw/                 -> Stores raw captured data (organized by person, letter, session).\n"
            "   - processed/           -> For cleaned & split data (train, val, test).\n"
            "   - docs/                -> Documentation and consent forms.\n"
            "   - short_images/        -> Contains images saved using a short path.\n\n"
            "Code Files:\n"
            " - data_collector.py      : Captures images and landmarks.\n"
            " - validate_dataset.py    : Validates dataset integrity.\n"
            " - augment_dataset.py     : Contains the augmentation pipeline.\n"
        )
        txt = scrolledtext.ScrolledText(self.setup_frame, width=80, height=15, wrap=tk.WORD, font=("Arial", 12))
        txt.insert(tk.END, instructions)
        txt.config(state=tk.DISABLED)
        txt.pack(pady=10)
        
        btn_create = ttk.Button(self.setup_frame, text="Create Project Structure", command=create_project_structure)
        btn_create.pack(pady=10)
    
    def create_collection_tab(self):
        header = ttk.Label(self.collection_frame, text="Data Collection", font=("Arial", 18, "bold"))
        header.pack(pady=10)
        
        instructions = (
            "Follow these steps to capture your sign language data:\n"
            "1. Enter the Person Name (e.g., Alice) and Session Name (e.g., MorningSession).\n"
            "2. Select the Arabic letter you are capturing and the number of samples to take.\n"
            "3. The live preview below shows what the camera sees. Ensure your hand is visible for landmark detection.\n"
            "4. Click 'Start Data Collection' to capture images and landmarks.\n"
        )
        lbl_instructions = ttk.Label(self.collection_frame, text=instructions, font=("Arial", 12), justify=tk.LEFT)
        lbl_instructions.pack(pady=5)
        
        frm = ttk.Frame(self.collection_frame)
        frm.pack(pady=10)
        
        ttk.Label(frm, text="Person Name:", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)
        self.person_name_var = tk.StringVar(value="Alice")
        ent_person = ttk.Entry(frm, textvariable=self.person_name_var, width=15, font=("Arial", 12))
        ent_person.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frm, text="Session Name:", font=("Arial", 12)).grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
        self.session_name_var = tk.StringVar(value="MorningSession")
        ent_session = ttk.Entry(frm, textvariable=self.session_name_var, width=15, font=("Arial", 12))
        ent_session.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(frm, text="Select Letter:", font=("Arial", 12)).grid(row=2, column=0, padx=5, pady=5, sticky=tk.E)
        self.letter_var = tk.StringVar(value="أ")
        letters = ["أ", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر",
                   "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف",
                   "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"]
        letter_menu = ttk.Combobox(frm, textvariable=self.letter_var, values=letters, state="readonly", width=5, font=("Arial", 12))
        letter_menu.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(frm, text="Number of Samples:", font=("Arial", 12)).grid(row=3, column=0, padx=5, pady=5, sticky=tk.E)
        self.samples_var = tk.IntVar(value=5)
        spin_samples = ttk.Spinbox(frm, from_=1, to=50, textvariable=self.samples_var, width=5, font=("Arial", 12))
        spin_samples.grid(row=3, column=1, padx=5, pady=5)
        
        btn_capture = ttk.Button(frm, text="Start Data Collection", command=self.run_data_collection)
        btn_capture.grid(row=4, column=0, columnspan=2, pady=10)
        
        self.preview_canvas = tk.Canvas(self.collection_frame, width=640, height=480, bg="black")
        self.preview_canvas.pack(pady=10)
        self.update_collection_preview()
    
    def update_collection_preview(self):
        ret, frame = self.cap_collection.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_canvas.imgtk = imgtk
            self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.preview_canvas.after(10, self.update_collection_preview)
    
    def run_data_collection(self):
        person_name = self.person_name_var.get().strip()
        session_name = self.session_name_var.get().strip()
        if not person_name or not session_name:
            messagebox.showerror("Error", "Please enter both a Person Name and a Session Name.")
            return
        letter = self.letter_var.get()
        num_samples = self.samples_var.get()
        # Use the fixed dataset directory
        dataset_dir = DATASET_ROOT
        
        # Create folder structure for landmarks (long path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = os.path.join(dataset_dir, "raw", person_name, letter, f"{session_name}_{timestamp}")
        landmarks_folder = os.path.join(session_folder, "landmarks")
        os.makedirs(landmarks_folder, exist_ok=True)
        # Images will be saved to the short_images folder (short relative path)
        images_folder = os.path.join(dataset_dir, "short_images")
        os.makedirs(images_folder, exist_ok=True)
        
        captured = 0
        for i in range(num_samples):
            ret, frame = self.cap_collection.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            
            # Define filename based on current sample index
            filename = f"{letter}_{person_name}_{session_name}_{timestamp}_{i:04d}"
            
            # 1. Save the original image to the short_images folder
            image_path = os.path.join(images_folder, f"{filename}.jpg")
            cv2.imwrite(image_path, frame)
            
            # 2. Process and save landmarks for the image to the landmarks folder
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
                np.save(os.path.join(landmarks_folder, f"{filename}_landmarks.npy"), landmarks)
            
            # 3. Apply augmentation and save the augmented image to the short_images folder
            augmented_frame = apply_augmentation(image=frame)
            augmented_image_path = os.path.join(images_folder, f"{filename}_augmented.jpg")
            cv2.imwrite(augmented_image_path, augmented_frame)
            
            # 4. Process and save landmarks for the augmented image
            rgb_augmented = cv2.cvtColor(augmented_frame, cv2.COLOR_BGR2RGB)
            results_aug = self.hands.process(rgb_augmented)
            if results_aug.multi_hand_landmarks:
                landmarks_aug = np.array([[lm.x, lm.y, lm.z] for lm in results_aug.multi_hand_landmarks[0].landmark])
                np.save(os.path.join(landmarks_folder, f"{filename}_augmented_landmarks.npy"), landmarks_aug)
            
            captured += 1
            time.sleep(0.1)
        
        messagebox.showinfo("Data Collection", f"Captured {captured} images for letter '{letter}'\nSaved in:\nLandmarks: {landmarks_folder}\nImages: {images_folder}")
    
    def create_validation_tab(self):
        header = ttk.Label(self.validation_frame, text="Dataset Validation", font=("Arial", 18, "bold"))
        header.pack(pady=10)
        
        instructions = (
            "This section runs a validation script to check your dataset's integrity.\n"
            "It verifies that each image is 1920x1080 and that each image has a corresponding landmarks file.\n"
            "Click 'Validate Dataset' to run the check and view any issues."
        )
        lbl = ttk.Label(self.validation_frame, text=instructions, font=("Arial", 12), justify=tk.LEFT)
        lbl.pack(pady=5)
        
        btn_validate = ttk.Button(self.validation_frame, text="Validate Dataset", command=self.run_validation)
        btn_validate.pack(pady=10)
        
        self.validation_output = scrolledtext.ScrolledText(self.validation_frame, width=80, height=15, font=("Arial", 12))
        self.validation_output.pack(pady=10)
    
    def run_validation(self):
        try:
            import validate_dataset
        except ImportError:
            messagebox.showerror("Error", "validate_dataset.py not found. Please create the project structure first.")
            return
        ds_path = DATASET_ROOT
        errors = validate_dataset.validate_dataset(ds_path)
        self.validation_output.delete(1.0, tk.END)
        if errors:
            self.validation_output.insert(tk.END, "Validation Errors:\n")
            for err in errors:
                self.validation_output.insert(tk.END, f"- {err}\n")
        else:
            self.validation_output.insert(tk.END, "Dataset is valid.")
    
    def create_augmentation_tab(self):
        header = ttk.Label(self.augmentation_frame, text="Data Augmentation Pipeline", font=("Arial", 18, "bold"))
        header.pack(pady=10)
        
        info = (
            "The augmentation pipeline (in augment_dataset.py) applies random transforms to your images:\n"
            " - RandomBrightnessContrast\n"
            " - HueSaturationValue\n"
            " - MotionBlur\n"
            " - ShiftScaleRotate\n\n"
            "This helps your model learn from varied data by simulating different conditions."
        )
        txt = scrolledtext.ScrolledText(self.augmentation_frame, width=80, height=15, font=("Arial", 12))
        txt.insert(tk.END, info)
        txt.config(state=tk.DISABLED)
        txt.pack(pady=10)
    
    def create_instructions_tab(self):
        header = ttk.Label(self.instructions_frame, text="Dataset Creation Manual", font=("Arial", 18, "bold"))
        header.pack(pady=10)
        
        manual = (
            "Step-by-Step Manual for Creating Your Sign Language Dataset\n\n"
            "1. **Project Setup**:\n"
            "   - Go to the 'Project Setup' tab and click 'Create Project Structure'.\n"
            "   - This creates all necessary folders and code files to organize your data.\n\n"
            "2. **Data Collection**:\n"
            "   - In the 'Data Collection' tab, enter your Person Name and Session Name.\n"
            "   - Select the Arabic letter you are capturing and the number of images to take.\n"
            "   - The live preview shows your camera feed; ensure your hand is visible for landmark detection.\n"
            "   - Click 'Start Data Collection' to capture data. Images are saved in the 'short_images' folder and landmarks in the 'raw' folder under your session.\n\n"
            "3. **Validation**:\n"
            "   - Use the 'Validation' tab to run checks on your dataset.\n"
            "   - The script ensures each image is 1920x1080 and that every image has a corresponding landmarks file.\n\n"
            "4. **Augmentation**:\n"
            "   - The 'Augmentation' tab displays the details of the augmentation pipeline, which applies random transforms to increase data diversity.\n\n"
            "5. **Understanding the Files & Folders**:\n"
            "   - **dataset/metadata.csv**: Logs details for each captured image.\n"
            "   - **dataset/raw/**: Contains your raw data organized by person, letter, and session (landmarks are stored here).\n"
            "   - **dataset/short_images/**: Contains the captured images (using a shorter relative path).\n"
            "   - **dataset/processed/**: For your cleaned and split data (train/val/test) after processing.\n"
            "   - **dataset/docs/ethical_consent_forms/**: Contains the digital consent form images.\n"
            "   - **data_collector.py**: Captures images and landmarks from your camera.\n"
            "   - **validate_dataset.py**: Validates the integrity of your dataset.\n"
            "   - **augment_dataset.py**: Defines the augmentation pipeline.\n\n"
            "Follow these steps in order to create a high-quality, well-organized dataset ready for AI training.\n"
            "If you have any questions, refer back to this manual for guidance."
        )
        txt = scrolledtext.ScrolledText(self.instructions_frame, width=90, height=30, font=("Arial", 12), wrap=tk.WORD)
        txt.insert(tk.END, manual)
        txt.config(state=tk.DISABLED)
        txt.pack(pady=10)

    def on_closing(self):
        if self.cap_collection.isOpened():
            self.cap_collection.release()
        self.root.destroy()

###############################################
# RUN THE GUI
###############################################

if __name__ == '__main__':
    root = tk.Tk()
    app = DatasetCreationGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

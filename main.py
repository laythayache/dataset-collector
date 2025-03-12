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

###############################################
# FUNCTIONS TO CREATE PROJECT STRUCTURE & FILES
###############################################

def create_project_structure():
    """
    Creates the project directory structure and code files.
    
    The folder structure created is:
      project_root/
        ├── dataset/
        │     ├── metadata.csv         -> Logs details for each captured image.
        │     ├── raw/                 -> Contains raw captured data.
        │     │      (organized as: person/letter/session/)
        │     ├── processed/           -> For cleaned and split data (train, val, test).
        │     │      ├── train/
        │     │      ├── val/
        │     │      └── test/
        │     └── docs/                -> Documentation (e.g., consent forms, guides).
        │            └── ethical_consent_forms/
        │
        ├── data_collector.py          -> Captures images and hand landmarks.
        ├── validate_dataset.py        -> Validates images and landmark files.
        └── augment_dataset.py         -> Defines a data augmentation pipeline.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_root = os.path.join(script_dir, "dataset")
        
        # Create dataset folder and subdirectories
        os.makedirs(dataset_root, exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "raw"), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "processed", "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "processed", "val"), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "processed", "test"), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, "docs", "ethical_consent_forms"), exist_ok=True)
        
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
        
        messagebox.showinfo("Success", "Project structure and files created successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create project structure:\n{str(e)}")

###############################################
# AUGMENTATION PIPELINE
###############################################

augmentation = Compose([
    RandomBrightnessContrast(p=0.5),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
    MotionBlur(blur_limit=7, p=0.3),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
], p=1.0)

def apply_augmentation(image):
    augmented = augmentation(image=image)
    print("Augmentation applied")  # Debug statement
    return augmented['image']

###############################################
# MAIN GUI: Dataset Creation Navigator
###############################################

class DatasetCreationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Creation Navigator")
        self.root.geometry("900x750")
        
        # Initialize MediaPipe for live preview
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Initialize camera for preview
        self.cap_collection = cv2.VideoCapture(0)
        if not self.cap_collection.isOpened():
            messagebox.showerror("Error", "Could not access the camera for preview!")
            self.root.destroy()
        
        # Create Notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')
        
        # Tabs
        self.setup_frame = ttk.Frame(self.notebook)
        self.collection_frame = ttk.Frame(self.notebook)
        self.validation_frame = ttk.Frame(self.notebook)
        self.augmentation_frame = ttk.Frame(self.notebook)
        self.instructions_frame = ttk.Frame(self.notebook)
        
        # Add Tabs to Notebook
        self.notebook.add(self.setup_frame, text="Project Setup")
        self.notebook.add(self.collection_frame, text="Data Collection")
        self.notebook.add(self.validation_frame, text="Validation")
        self.notebook.add(self.augmentation_frame, text="Augmentation")
        self.notebook.add(self.instructions_frame, text="Instructions")
        
        # Create tab content
        self.create_setup_tab()
        self.create_collection_tab()
        self.create_validation_tab()
        self.create_augmentation_tab()
        self.create_instructions_tab()

    ###################################################
    # Project Setup Tab
    ###################################################
    def create_setup_tab(self):
        lbl = ttk.Label(self.setup_frame, text="Project Setup", font=("Arial", 18, "bold"))
        lbl.pack(pady=10)
        
        instructions = (
            "This section will create the necessary project structure and code files for your sign language dataset.\n\n"
            "Folder Structure:\n"
            " - dataset/\n"
            "   - raw/\n"
            "   - processed/\n"
            "   - docs/\n\n"
            "Code Files:\n"
            " - data_collector.py\n"
            " - validate_dataset.py\n"
            " - augment_dataset.py\n"
        )
        txt = scrolledtext.ScrolledText(self.setup_frame, width=80, height=15, wrap=tk.WORD, font=("Arial", 12))
        txt.insert(tk.END, instructions)
        txt.config(state=tk.DISABLED)
        txt.pack(pady=10)
        
        btn_create = ttk.Button(self.setup_frame, text="Create Project Structure", command=create_project_structure)
        btn_create.pack(pady=10)
    
    ###################################################
    # Data Collection Tab
    ###################################################
    def create_collection_tab(self):
        header = ttk.Label(self.collection_frame, text="Data Collection", font=("Arial", 18, "bold"))
        header.pack(pady=10)
        
        instructions = (
            "Follow these steps to capture your sign language data:\n"
            "1. Enter the Person Name (e.g., Alice) and Session Name (e.g., MorningSession).\n"
            "2. Select the Arabic letter and the number of samples.\n"
            "3. Click 'Start Data Collection' to capture images and landmarks.\n"
        )
        lbl_instructions = ttk.Label(self.collection_frame, text=instructions, font=("Arial", 12), justify=tk.LEFT)
        lbl_instructions.pack(pady=5)
        
        frm = ttk.Frame(self.collection_frame)
        frm.pack(pady=10)
        
        ttk.Label(frm, text="Person Name:", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)
        self.person_name_var = tk.StringVar(value="Alice")
        ttk.Entry(frm, textvariable=self.person_name_var, width=15, font=("Arial", 12)).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frm, text="Session Name:", font=("Arial", 12)).grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
        self.session_name_var = tk.StringVar(value="MorningSession")
        ttk.Entry(frm, textvariable=self.session_name_var, width=15, font=("Arial", 12)).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(frm, text="Select Letter:", font=("Arial", 12)).grid(row=2, column=0, padx=5, pady=5, sticky=tk.E)
        self.letter_var = tk.StringVar(value="أ")
        letters = [
            "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20",
            "21","22","23","24","25","26","27","28"
        ]
        letter_menu = ttk.Combobox(frm, textvariable=self.letter_var, values=letters, state="readonly", width=5, font=("Arial", 12))
        letter_menu.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(frm, text="Number of Samples:", font=("Arial", 12)).grid(row=3, column=0, padx=5, pady=5, sticky=tk.E)
        self.samples_var = tk.IntVar(value=5)
        ttk.Spinbox(frm, from_=1, to=50, textvariable=self.samples_var, width=5, font=("Arial", 12)).grid(row=3, column=1, padx=5, pady=5)
        
        btn_capture = ttk.Button(frm, text="Start Data Collection", command=self.run_data_collection)
        btn_capture.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Camera preview
        self.preview_canvas = tk.Canvas(self.collection_frame, width=640, height=480, bg="black")
        self.preview_canvas.pack(pady=10)
        self.update_collection_preview()
    
    def update_collection_preview(self):
        """Continuously read from self.cap_collection to update the live preview canvas."""
        ret, frame = self.cap_collection.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.preview_canvas.imgtk = imgtk
            self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.preview_canvas.after(10, self.update_collection_preview)
    
    def run_data_collection(self):
        """Called when user presses 'Start Data Collection'."""
        person_name = self.person_name_var.get().strip()
        session_name = self.session_name_var.get().strip()
        letter = self.letter_var.get()
        num_samples = self.samples_var.get()
        
        if not person_name or not session_name:
            messagebox.showerror("Error", "Please enter both a Person Name and a Session Name.")
            return
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(script_dir, "dataset")
        
        # Build final session path: dataset/raw/{person_name}/{letter}/{session_name_timestamp}/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_subfolder = f"{session_name}_{timestamp}"
        
        # e.g. c:\...\dataset\raw\Alice\أ\MorningSession_20250312_120000\
        base_session_folder = os.path.join(dataset_dir, "raw", person_name, letter, session_subfolder)
        images_folder = os.path.join(base_session_folder, "images")
        landmarks_folder = os.path.join(base_session_folder, "landmarks")
        
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(landmarks_folder, exist_ok=True)
        
        # We'll capture frames from the existing preview camera (self.cap_collection).
        # If you see many ret=False, consider opening a new cap for capturing.
        captured = 0
        for i in range(num_samples):
            ret, frame = self.cap_collection.read()
            if not ret:
                print(f"[WARNING] No frame from camera at iteration {i}. Skipping.")
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Build the image filename
            filename = f"{letter}_{person_name}_{session_name}_{timestamp}_{i:04d}"
            image_path = os.path.join(images_folder, f"{filename}.jpg")
            
            # Save the original image
            success_img = cv2.imwrite(image_path, frame)
            print("[DEBUG] Saved original:", success_img, "->", image_path)
            
            # If we have hand landmarks, save them
            if results.multi_hand_landmarks:
                # The first (or only) hand's 3D coords
                first_hand = results.multi_hand_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in first_hand.landmark])
                landmark_path = os.path.join(landmarks_folder, f"{filename}_landmarks.npy")
                np.save(landmark_path, landmarks)
                print("[DEBUG] Saved landmarks ->", landmark_path)
            
            # Apply augmentation and save augmented image
            augmented_frame = apply_augmentation(image=frame)
            augmented_image_path = os.path.join(images_folder, f"{filename}_augmented.jpg")
            success_aug = cv2.imwrite(augmented_image_path, augmented_frame)
            print("[DEBUG] Saved augmented:", success_aug, "->", augmented_image_path)
            
            captured += 1
            time.sleep(0.1)
        
        messagebox.showinfo(
            "Data Collection",
            f"Captured {captured} images for letter '{letter}'\nSaved in:\n{base_session_folder}"
        )
    
    ###################################################
    # Validation Tab
    ###################################################
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
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ds_path = os.path.join(script_dir, "dataset")
        
        errors = validate_dataset.validate_dataset(ds_path)
        self.validation_output.delete(1.0, tk.END)
        
        if errors:
            self.validation_output.insert(tk.END, "Validation Errors:\n")
            for err in errors:
                self.validation_output.insert(tk.END, f"- {err}\n")
        else:
            self.validation_output.insert(tk.END, "Dataset is valid.")
    
    ###################################################
    # Augmentation Tab
    ###################################################
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
    
    ###################################################
    # Instructions Tab
    ###################################################
    def create_instructions_tab(self):
        header = ttk.Label(self.instructions_frame, text="Dataset Creation Manual", font=("Arial", 18, "bold"))
        header.pack(pady=10)
        
        manual = (
            "Step-by-Step Manual:\n\n"
            "1. **Project Setup**:\n"
            "   - Use 'Create Project Structure' to generate folders.\n\n"
            "2. **Data Collection**:\n"
            "   - Enter Person Name, Session Name, Letter, and number of images.\n"
            "   - Click 'Start Data Collection'. Images go into 'images/' and landmarks go into 'landmarks/'.\n\n"
            "3. **Validation**:\n"
            "   - Check resolution and that each image has a corresponding landmarks file.\n\n"
            "4. **Augmentation**:\n"
            "   - Additional transforms can help your model generalize better.\n"
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

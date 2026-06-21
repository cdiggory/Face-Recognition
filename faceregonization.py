import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import os
import csv
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ImageTk
import threading
import time
import warnings
warnings.filterwarnings("ignore")

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("Warning: face_recognition not installed. Using fallback mode.")

try:
    from sklearn.neighbors import KNeighborsClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Using simplified mode.")

CONFIG = {
    "dataset_dir": "face_dataset",
    "attendance_file": "attendance.csv",
    "model_file": "knn_model.pkl",
    "students_file": "students.json",
    "knn_neighbors": 3,
    "tolerance": 0.5,
    "captures_per_student": 8,
    "camera_index": 0,
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_students():
    if os.path.exists(CONFIG["students_file"]):
        with open(CONFIG["students_file"], "r") as f:
            return json.load(f)
    return []

def save_students(students):
    with open(CONFIG["students_file"], "w") as f:
        json.dump(students, f, indent=2)

def generate_student_id():
    students = load_students()
    if not students:
        return "STU001"
    max_id = max([int(s["id"][3:]) for s in students])
    return f"STU{max_id + 1:03d}"

def create_synthetic_face(person_id, variation):
    np.random.seed(person_id * 100 + variation)
    img = np.ones((128, 128, 3), dtype=np.uint8) * 220
    cx, cy = 64, 64
    color = (int(180 + person_id * 3) % 256, int(140 + person_id * 7) % 256, int(100 + person_id * 5) % 256)
    cv2.ellipse(img, (cx, cy), (35, 45), 0, 0, 360, color, -1)
    eye_y = cy - 10
    cv2.circle(img, (cx - 12, eye_y), 5, (50, 50, 50), -1)
    cv2.circle(img, (cx + 12, eye_y), 5, (50, 50, 50), -1)
    cv2.circle(img, (cx, cy + 5), 3, (160, 100, 80), -1)
    cv2.ellipse(img, (cx, cy + 20), (10, 5), 0, 0, 180, (120, 60, 60), 2)
    shift = variation % 5
    M = np.float32([[1, 0, shift], [0, 1, shift]])
    img = cv2.warpAffine(img, M, (128, 128))
    return img

def get_face_encoding(image):
    if not FACE_REC_AVAILABLE:
        return None
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image
    encodings = face_recognition.face_encodings(rgb)
    if encodings:
        return encodings[0]
    return None

class AttendanceManager:
    def __init__(self):
        ensure_dir(CONFIG["dataset_dir"])
        self.students = load_students()
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.attendance_cache = {}
        self.load_attendance_cache()

    def load_attendance_cache(self):
        if os.path.exists(CONFIG["attendance_file"]):
            df = pd.read_csv(CONFIG["attendance_file"])
            for _, row in df.iterrows():
                key = f"{row['Date']}_{row['StudentID']}"
                self.attendance_cache[key] = row.to_dict()

    def mark_attendance(self, student_id, student_name, status="Present"):
        key = f"{self.today}_{student_id}"
        if key in self.attendance_cache:
            return False, "Already marked today"
        record = {
            "Date": self.today,
            "Time": datetime.now().strftime("%H:%M:%S"),
            "StudentID": student_id,
            "Name": student_name,
            "Status": status
        }
        self.attendance_cache[key] = record
        file_exists = os.path.exists(CONFIG["attendance_file"])
        with open(CONFIG["attendance_file"], "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Date", "Time", "StudentID", "Name", "Status"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
        return True, "Marked successfully"

    def get_today_attendance(self):
        return [v for k, v in self.attendance_cache.items() if k.startswith(self.today)]

    def get_all_attendance(self):
        if os.path.exists(CONFIG["attendance_file"]):
            return pd.read_csv(CONFIG["attendance_file"])
        return pd.DataFrame()

class FaceRecognitionEngine:
    def __init__(self):
        self.knn = None
        self.encodings = {}
        self.student_ids = []
        self.trained = False
        self.load_model()

    def load_model(self):
        if os.path.exists(CONFIG["model_file"]):
            with open(CONFIG["model_file"], "rb") as f:
                data = pickle.load(f)
                self.knn = data.get("knn")
                self.encodings = data.get("encodings", {})
                self.student_ids = data.get("student_ids", [])
                self.trained = data.get("trained", False)

    def save_model(self):
        data = {"knn": self.knn, "encodings": self.encodings, "student_ids": self.student_ids, "trained": self.trained}
        with open(CONFIG["model_file"], "wb") as f:
            pickle.dump(data, f)

    def train_model(self, progress_callback=None):
        students = load_students()
        if not students:
            return False, "No students registered"
        X, y, student_ids = [], [], []
        total = len(students)
        for idx, student in enumerate(students):
            student_dir = os.path.join(CONFIG["dataset_dir"], student["id"])
            if not os.path.exists(student_dir):
                continue
            images = []
            for fname in os.listdir(student_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = cv2.imread(os.path.join(student_dir, fname))
                    if img is not None:
                        images.append(img)
            needed = max(0, CONFIG["captures_per_student"] - len(images))
            for v in range(needed):
                images.append(create_synthetic_face(idx, v))
            enc_count = 0
            for img in images:
                enc = get_face_encoding(img)
                if enc is not None:
                    X.append(enc)
                    y.append(student["name"])
                    student_ids.append(student["id"])
                    enc_count += 1
                else:
                    vec = np.random.randn(128)
                    vec = vec / (np.linalg.norm(vec) + 1e-9)
                    X.append(vec)
                    y.append(student["name"])
                    student_ids.append(student["id"])
                    enc_count += 1
            if progress_callback:
                progress_callback(idx + 1, total, student["name"], enc_count)
        if len(X) < CONFIG["knn_neighbors"]:
            return False, f"Need at least {CONFIG['knn_neighbors']} samples, got {len(X)}"
        X = np.array(X)
        if SKLEARN_AVAILABLE:
            self.knn = KNeighborsClassifier(n_neighbors=min(CONFIG["knn_neighbors"], len(X)), metric="euclidean", weights="distance", algorithm="ball_tree")
            self.knn.fit(X, y)
        else:
            self.encodings = {sid: enc for sid, enc in zip(student_ids, X)}
            self.student_ids = student_ids
        self.trained = True
        self.save_model()
        return True, f"Model trained on {len(X)} samples"

    def predict(self, image):
        if not self.trained:
            return None, 0.0
        enc = get_face_encoding(image)
        if enc is None:
            return None, 0.0
        enc_vec = np.array([enc])
        if SKLEARN_AVAILABLE and self.knn is not None:
            try:
                distances, indices = self.knn.kneighbors(enc_vec, n_neighbors=1)
                pred = self.knn.predict(enc_vec)[0]
                confidence = 1.0 / (1.0 + distances[0][0])
                students = load_students()
                for s in students:
                    if s["name"] == pred:
                        return s, confidence
                return None, 0.0
            except Exception:
                return None, 0.0
        else:
            min_dist = float('inf')
            best_id = None
            for sid, ref_enc in self.encodings.items():
                dist = np.linalg.norm(enc - ref_enc)
                if dist < min_dist:
                    min_dist = dist
                    best_id = sid
            if best_id and min_dist < CONFIG["tolerance"] * 100:
                students = load_students()
                for s in students:
                    if s["id"] == best_id:
                        return s, 1.0 / (1.0 + min_dist)
            return None, 0.0

class CameraManager:
    def __init__(self, canvas, width=640, height=480):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.current_frame = None
        self.thread = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(CONFIG["camera_index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.canvas.delete("all")

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.after(0, self._update_canvas, imgtk)
            time.sleep(0.03)

    def _update_canvas(self, imgtk):
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def get_frame(self):
        return self.current_frame

class SmartAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System - Face Recognition")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2e")
        self.attendance_mgr = AttendanceManager()
        self.face_engine = FaceRecognitionEngine()
        self.camera = None
        self.capture_count = 0
        self.current_student = None
        self.is_capturing = False
        self.auto_recognize = False
        self.setup_styles()
        self.build_header()
        self.build_sidebar()
        self.build_main_content()
        self.build_status_bar()
        self.show_dashboard()
        self.check_dependencies()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        bg_dark = "#1e1e2e"
        bg_card = "#2d2d44"
        accent = "#89b4fa"
        accent_hover = "#b4befe"
        text_primary = "#cdd6f4"
        text_secondary = "#a6adc8"
        success = "#a6e3a1"
        warning = "#f9e2af"
        danger = "#f38ba8"
        style.configure("Dark.TFrame", background=bg_dark)
        style.configure("Card.TFrame", background=bg_card)
        style.configure("Dark.TLabel", background=bg_dark, foreground=text_primary, font=("Segoe UI", 11))
        style.configure("Card.TLabel", background=bg_card, foreground=text_primary, font=("Segoe UI", 11))
        style.configure("Title.TLabel", background=bg_dark, foreground=accent, font=("Segoe UI", 24, "bold"))
        style.configure("Subtitle.TLabel", background=bg_dark, foreground=text_secondary, font=("Segoe UI", 12))
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), foreground=bg_dark, background=accent)
        style.map("Accent.TButton", background=[("active", accent_hover), ("pressed", accent)])
        style.configure("Secondary.TButton", font=("Segoe UI", 10), foreground=text_primary, background=bg_card)
        style.map("Secondary.TButton", background=[("active", "#3d3d5c"), ("pressed", bg_card)])
        style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), foreground=bg_dark, background=danger)
        style.map("Danger.TButton", background=[("active", "#f5c2e7"), ("pressed", danger)])
        style.configure("Success.TButton", font=("Segoe UI", 10, "bold"), foreground=bg_dark, background=success)
        style.map("Success.TButton", background=[("active", "#94e2d5"), ("pressed", success)])
        style.configure("Dark.TEntry", fieldbackground=bg_card, foreground=text_primary, insertcolor=text_primary)
        style.configure("Dark.Horizontal.TProgressbar", background=accent, troughcolor=bg_card)

    def build_header(self):
        header = ttk.Frame(self.root, style="Dark.TFrame")
        header.pack(fill=tk.X, padx=0, pady=0)
        logo_frame = ttk.Frame(header, style="Dark.TFrame")
        logo_frame.pack(side=tk.LEFT, padx=20, pady=15)
        ttk.Label(logo_frame, text="FACE RECOGNITION", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(logo_frame, text="Smart Attendance System", style="Subtitle.TLabel").pack(anchor=tk.W)
        self.clock_label = ttk.Label(header, text="", style="Dark.TLabel", font=("Segoe UI", 14))
        self.clock_label.pack(side=tk.RIGHT, padx=20, pady=15)
        self.update_clock()

    def update_clock(self):
        now = datetime.now().strftime("%H:%M:%S  |  %d %b %Y")
        self.clock_label.config(text=now)
        self.root.after(1000, self.update_clock)

    def build_sidebar(self):
        sidebar = ttk.Frame(self.root, style="Card.TFrame", width=220)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar.pack_propagate(False)
        menu_items = [
            ("Dashboard", self.show_dashboard, "Dashboard"),
            ("Register Face", self.show_register, "Register"),
            ("Mark Attendance", self.show_attendance, "Attendance"),
            ("View Records", self.show_records, "Records"),
            ("Students", self.show_students, "Students"),
            ("Train Model", self.show_train, "Train"),
            ("Settings", self.show_settings, "Settings"),
        ]
        ttk.Label(sidebar, text="MENU", style="Card.TLabel", font=("Segoe UI", 9, "bold"), foreground="#6c7086").pack(anchor=tk.W, padx=20, pady=(20, 10))
        self.menu_buttons = {}
        for text, command, short in menu_items:
            btn = tk.Button(sidebar, text=f"  {short}", font=("Segoe UI", 11), bg="#2d2d44", fg="#cdd6f4", activebackground="#3d3d5c", activeforeground="#cdd6f4", bd=0, padx=15, pady=10, anchor=tk.W, cursor="hand2", command=command)
            btn.pack(fill=tk.X, padx=10, pady=2)
            self.menu_buttons[text] = btn
        ttk.Label(sidebar, text="", style="Card.TLabel").pack(expand=True)
        stats_frame = ttk.Frame(sidebar, style="Card.TFrame")
        stats_frame.pack(fill=tk.X, padx=15, pady=15)
        self.stat_total_students = ttk.Label(stats_frame, text="Students: 0", style="Card.TLabel")
        self.stat_total_students.pack(anchor=tk.W)
        self.stat_today_present = ttk.Label(stats_frame, text="Present Today: 0", style="Card.TLabel")
        self.stat_today_present.pack(anchor=tk.W)
        self.update_stats()

    def update_stats(self):
        students = load_students()
        self.stat_total_students.config(text=f"Students: {len(students)}")
        today_records = self.attendance_mgr.get_today_attendance()
        present = len([r for r in today_records if r["Status"] == "Present"])
        self.stat_today_present.config(text=f"Present Today: {present}")

    def set_active_menu(self, active_text):
        for text, btn in self.menu_buttons.items():
            if text == active_text:
                btn.config(bg="#89b4fa", fg="#1e1e2e")
            else:
                btn.config(bg="#2d2d44", fg="#cdd6f4")

    def build_main_content(self):
        self.main_frame = ttk.Frame(self.root, style="Dark.TFrame")
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

    def clear_main(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def build_status_bar(self):
        self.status_bar = ttk.Frame(self.root, style="Card.TFrame", height=30)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar.pack_propagate(False)
        self.status_label = ttk.Label(self.status_bar, text="Ready", style="Card.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=15, pady=5)
        self.model_status = ttk.Label(self.status_bar, text="Model: Not Trained", style="Card.TLabel", foreground="#f38ba8")
        self.model_status.pack(side=tk.RIGHT, padx=15, pady=5)
        if self.face_engine.trained:
            self.model_status.config(text="Model: Trained", foreground="#a6e3a1")

    def set_status(self, text, color="#cdd6f4"):
        self.status_label.config(text=text, foreground=color)

    def check_dependencies(self):
        if not FACE_REC_AVAILABLE:
            messagebox.showwarning("Missing Dependency", "face_recognition library not found.\n\nInstall with:\npip install face-recognition\n\nThe app will run in fallback mode with synthetic faces.")
        if not SKLEARN_AVAILABLE:
            messagebox.showwarning("Missing Dependency", "scikit-learn not found.\n\nInstall with:\npip install scikit-learn\n\nThe app will use simplified recognition.")

    def show_dashboard(self):
        self.set_active_menu("Dashboard")
        self.clear_main()
        self.stop_camera()
        ttk.Label(self.main_frame, text="Dashboard", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(self.main_frame, text="Overview of attendance system", style="Subtitle.TLabel").pack(anchor=tk.W, pady=(0, 20))
        cards = ttk.Frame(self.main_frame, style="Dark.TFrame")
        cards.pack(fill=tk.X, pady=10)
        students = load_students()
        today_records = self.attendance_mgr.get_today_attendance()
        present_count = len([r for r in today_records if r["Status"] == "Present"])
        absent_count = len(students) - present_count
        model_status = "Trained" if self.face_engine.trained else "Not Trained"
        self.create_card(cards, "Total Students", str(len(students)), "#89b4fa").pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.create_card(cards, "Present Today", str(present_count), "#a6e3a1").pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.create_card(cards, "Absent Today", str(absent_count), "#f38ba8").pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.create_card(cards, "Model Status", model_status, "#f9e2af").pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        ttk.Label(self.main_frame, text="Recent Activity", style="Title.TLabel", font=("Segoe UI", 16, "bold")).pack(anchor=tk.W, pady=(30, 10))
        activity_frame = ttk.Frame(self.main_frame, style="Card.TFrame")
        activity_frame.pack(fill=tk.BOTH, expand=True)
        columns = ("Time", "Name", "Status")
        tree = ttk.Treeview(activity_frame, columns=columns, show="headings", height=10)
        tree.heading("Time", text="Time")
        tree.heading("Name", text="Name")
        tree.heading("Status", text="Status")
        tree.column("Time", width=150)
        tree.column("Name", width=300)
        tree.column("Status", width=150)
        scrollbar = ttk.Scrollbar(activity_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        for record in reversed(today_records[-20:]):
            tree.insert("", tk.END, values=(record["Time"], record["Name"], record["Status"]))

    def create_card(self, parent, title, value, color):
        card = tk.Frame(parent, bg="#2d2d44", bd=0, highlightbackground="#3d3d5c", highlightthickness=1, padx=20, pady=20)
        tk.Label(card, text=title, font=("Segoe UI", 11), bg="#2d2d44", fg="#a6adc8").pack(anchor=tk.W)
        tk.Label(card, text=value, font=("Segoe UI", 28, "bold"), bg="#2d2d44", fg=color).pack(anchor=tk.W)
        return card

    def show_register(self):
        self.set_active_menu("Register Face")
        self.clear_main()
        self.stop_camera()
        ttk.Label(self.main_frame, text="Register New Student", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(self.main_frame, text="Capture face images for recognition", style="Subtitle.TLabel").pack(anchor=tk.W, pady=(0, 20))
        content = ttk.Frame(self.main_frame, style="Dark.TFrame")
        content.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(content, style="Dark.TFrame", width=350)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left.pack_propagate(False)
        form = ttk.Frame(left, style="Card.TFrame", padding=20)
        form.pack(fill=tk.X, pady=10)
        ttk.Label(form, text="Student Details", style="Card.TLabel", font=("Segoe UI", 14, "bold")).pack(anchor=tk.W, pady=(0, 15))
        ttk.Label(form, text="Student ID", style="Card.TLabel").pack(anchor=tk.W)
        self.reg_id_var = tk.StringVar(value=generate_student_id())
        ttk.Entry(form, textvariable=self.reg_id_var, state="readonly", style="Dark.TEntry").pack(fill=tk.X, pady=(0, 10))
        ttk.Label(form, text="Full Name *", style="Card.TLabel").pack(anchor=tk.W)
        self.reg_name_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.reg_name_var, style="Dark.TEntry").pack(fill=tk.X, pady=(0, 10))
        ttk.Label(form, text="Roll Number", style="Card.TLabel").pack(anchor=tk.W)
        self.reg_roll_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.reg_roll_var, style="Dark.TEntry").pack(fill=tk.X, pady=(0, 10))
        ttk.Label(form, text="Email", style="Card.TLabel").pack(anchor=tk.W)
        self.reg_email_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.reg_email_var, style="Dark.TEntry").pack(fill=tk.X, pady=(0, 10))
        ttk.Label(form, text=f"Captures Needed: {CONFIG['captures_per_student']}", style="Card.TLabel").pack(anchor=tk.W, pady=(10, 0))
        self.capture_progress = ttk.Progressbar(form, mode="determinate", maximum=CONFIG["captures_per_student"], style="Dark.Horizontal.TProgressbar")
        self.capture_progress.pack(fill=tk.X, pady=5)
        self.capture_status = ttk.Label(form, text="0 / {} captured".format(CONFIG["captures_per_student"]), style="Card.TLabel")
        self.capture_status.pack(anchor=tk.W)
        btn_frame = ttk.Frame(form, style="Card.TFrame")
        btn_frame.pack(fill=tk.X, pady=(20, 0))
        ttk.Button(btn_frame, text="Start Camera", command=self.start_register_camera, style="Accent.TButton").pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_frame, text="Capture Photo", command=self.capture_photo, style="Success.TButton").pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_frame, text="Save Student", command=self.save_student, style="Secondary.TButton").pack(fill=tk.X, pady=(0, 5))
        ttk.Button(btn_frame, text="Reset", command=self.reset_register, style="Danger.TButton").pack(fill=tk.X)
        right = ttk.Frame(content, style="Card.TFrame")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.reg_canvas = tk.Canvas(right, width=640, height=480, bg="#181825", highlightthickness=0)
        self.reg_canvas.pack(padx=20, pady=20)
        instructions = ttk.Frame(right, style="Card.TFrame")
        instructions.pack(fill=tk.X, padx=20, pady=(0, 20))
        ttk.Label(instructions, text="Instructions:", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        tips = ["1. Ensure good lighting on your face", "2. Look directly at the camera", "3. Keep face centered in the frame", "4. Vary angle slightly between captures", "5. Press 'Capture' to take photo", f"6. Need {CONFIG['captures_per_student']} clear face images"]
        for tip in tips:
            ttk.Label(instructions, text=tip, style="Card.TLabel", font=("Segoe UI", 10)).pack(anchor=tk.W)
        self.capture_count = 0
        self.captured_images = []

    def start_register_camera(self):
        if not self.reg_name_var.get().strip():
            messagebox.showerror("Error", "Please enter student name first")
            return
        self.camera = CameraManager(self.reg_canvas, 640, 480)
        self.camera.start()
        self.set_status("Camera started for registration")
        self.is_capturing = True

    def capture_photo(self):
        if not self.is_capturing or not self.camera or self.camera.current_frame is None:
            return
        frame = self.camera.current_frame.copy()
        self.captured_images.append(frame)
        self.capture_count += 1
        self.capture_progress["value"] = self.capture_count
        self.capture_status.config(text=f"{self.capture_count} / {CONFIG['captures_per_student']} captured")
        self.reg_canvas.config(bg="white")
        self.root.after(100, lambda: self.reg_canvas.config(bg="#181825"))
        self.set_status(f"Captured photo {self.capture_count}")
        if self.capture_count >= CONFIG["captures_per_student"]:
            self.set_status("All captures complete! Click Save Student", "#a6e3a1")
            messagebox.showinfo("Complete", f"All {CONFIG['captures_per_student']} photos captured!")

    def save_student(self):
        name = self.reg_name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter student name")
            return
        if self.capture_count < CONFIG["captures_per_student"]:
            msg = f"Only {self.capture_count} images captured. Need {CONFIG['captures_per_student']}. Continue anyway?"
            if not messagebox.askyesno("Confirm", msg):
                return
        student_id = self.reg_id_var.get()
        student_dir = os.path.join(CONFIG["dataset_dir"], student_id)
        ensure_dir(student_dir)
        for i, img in enumerate(self.captured_images):
            path = os.path.join(student_dir, f"{student_id}_{i}.jpg")
            cv2.imwrite(path, img)
        students = load_students()
        students.append({"id": student_id, "name": name, "roll": self.reg_roll_var.get(), "email": self.reg_email_var.get(), "registered": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "images": len(self.captured_images)})
        save_students(students)
        self.stop_camera()
        self.set_status(f"Student {name} registered successfully!", "#a6e3a1")
        self.update_stats()
        messagebox.showinfo("Success", f"Student '{name}' registered successfully!\n\nPlease train the model before marking attendance.")
        self.reset_register()

    def reset_register(self):
        self.stop_camera()
        self.reg_id_var.set(generate_student_id())
        self.reg_name_var.set("")
        self.reg_roll_var.set("")
        self.reg_email_var.set("")
        self.capture_count = 0
        self.captured_images = []
        self.capture_progress["value"] = 0
        self.capture_status.config(text=f"0 / {CONFIG['captures_per_student']} captured")
        self.is_capturing = False

    def show_attendance(self):
        self.set_active_menu("Mark Attendance")
        self.clear_main()
        self.stop_camera()
        ttk.Label(self.main_frame, text="Mark Attendance", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(self.main_frame, text="Face recognition based attendance marking", style="Subtitle.TLabel").pack(anchor=tk.W, pady=(0, 20))
        if not self.face_engine.trained:
            warning = ttk.Frame(self.main_frame, style="Card.TFrame")
            warning.pack(fill=tk.X, pady=10)
            ttk.Label(warning, text="Model not trained! Please train the model first in the Train Model section.", style="Card.TLabel", foreground="#f38ba8", font=("Segoe UI", 11, "bold")).pack(padx=20, pady=15)
        content = ttk.Frame(self.main_frame, style="Dark.TFrame")
        content.pack(fill=tk.BOTH, expand=True)
        cam_frame = ttk.Frame(content, style="Card.TFrame")
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.att_canvas = tk.Canvas(cam_frame, width=640, height=480, bg="#181825", highlightthickness=0)
        self.att_canvas.pack(padx=20, pady=20)
        btn_frame = ttk.Frame(cam_frame, style="Card.TFrame")
        btn_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        ttk.Button(btn_frame, text="Start Camera", command=self.start_attendance_camera, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Stop Camera", command=self.stop_camera, style="Danger.TButton").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Manual Mark", command=self.manual_mark, style="Secondary.TButton").pack(side=tk.LEFT)
        right = ttk.Frame(content, style="Dark.TFrame", width=350)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(20, 0))
        right.pack_propagate(False)
        result_frame = ttk.Frame(right, style="Card.TFrame", padding=20)
        result_frame.pack(fill=tk.X, pady=10)
        ttk.Label(result_frame, text="Recognition Result", style="Card.TLabel", font=("Segoe UI", 14, "bold")).pack(anchor=tk.W, pady=(0, 15))
        self.rec_name = ttk.Label(result_frame, text="—", style="Card.TLabel", font=("Segoe UI", 18, "bold"))
        self.rec_name.pack(anchor=tk.W)
        self.rec_id = ttk.Label(result_frame, text="ID: —", style="Card.TLabel")
        self.rec_id.pack(anchor=tk.W, pady=5)
        self.rec_confidence = ttk.Label(result_frame, text="Confidence: —", style="Card.TLabel")
        self.rec_confidence.pack(anchor=tk.W)
        self.rec_status = ttk.Label(result_frame, text="Status: Waiting...", style="Card.TLabel", foreground="#f9e2af", font=("Segoe UI", 12, "bold"))
        self.rec_status.pack(anchor=tk.W, pady=(10, 0))
        ttk.Label(right, text="Today's Attendance", style="Dark.TLabel", font=("Segoe UI", 14, "bold")).pack(anchor=tk.W, pady=(20, 10))
        list_frame = ttk.Frame(right, style="Card.TFrame")
        list_frame.pack(fill=tk.BOTH, expand=True)
        self.att_listbox = tk.Listbox(list_frame, bg="#2d2d44", fg="#cdd6f4", font=("Segoe UI", 11), bd=0, highlightthickness=0, selectbackground="#89b4fa", selectforeground="#1e1e2e")
        self.att_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.att_listbox.yview)
        self.att_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        self.refresh_attendance_list()
        self.auto_recognize = False

    def start_attendance_camera(self):
        if not self.face_engine.trained:
            messagebox.showwarning("Warning", "Model not trained! Please train first.")
            return
        self.camera = CameraManager(self.att_canvas, 640, 480)
        self.camera.start()
        self.auto_recognize = True
        self.set_status("Attendance camera started - Auto recognition enabled")
        self.recognition_loop()

    def recognition_loop(self):
        if not self.auto_recognize or not self.camera or not self.camera.running:
            return
        frame = self.camera.get_frame()
        if frame is not None:
            student, confidence = self.face_engine.predict(frame)
            if student and confidence > 0.3:
                self.rec_name.config(text=student["name"], foreground="#a6e3a1")
                self.rec_id.config(text=f"ID: {student['id']}")
                self.rec_confidence.config(text=f"Confidence: {confidence:.2%}")
                success, msg = self.attendance_mgr.mark_attendance(student["id"], student["name"])
                if success:
                    self.rec_status.config(text="Marked PRESENT", foreground="#a6e3a1")
                    self.set_status(f"Attendance marked for {student['name']}", "#a6e3a1")
                    self.refresh_attendance_list()
                    self.update_stats()
                else:
                    self.rec_status.config(text=f"Already marked", foreground="#f9e2af")
            else:
                self.rec_name.config(text="Unknown", foreground="#f38ba8")
                self.rec_id.config(text="ID: —")
                self.rec_confidence.config(text="Confidence: —")
                self.rec_status.config(text="No face detected", foreground="#6c7086")
        self.root.after(1000, self.recognition_loop)

    def refresh_attendance_list(self):
        self.att_listbox.delete(0, tk.END)
        today = self.attendance_mgr.get_today_attendance()
        for record in today:
            status_icon = "P" if record["Status"] == "Present" else "A"
            self.att_listbox.insert(tk.END, f"[{status_icon}] {record['Name']} ({record['Time']})")

    def manual_mark(self):
        students = load_students()
        if not students:
            messagebox.showerror("Error", "No students registered")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Manual Attendance")
        dialog.geometry("400x500")
        dialog.configure(bg="#1e1e2e")
        ttk.Label(dialog, text="Select Student", style="Dark.TLabel", font=("Segoe UI", 14, "bold")).pack(pady=20)
        listbox = tk.Listbox(dialog, bg="#2d2d44", fg="#cdd6f4", font=("Segoe UI", 12), bd=0, highlightthickness=0, selectbackground="#89b4fa")
        listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        for s in students:
            listbox.insert(tk.END, f"{s['name']} ({s['id']})")
        def mark():
            sel = listbox.curselection()
            if not sel:
                return
            student = students[sel[0]]
            success, msg = self.attendance_mgr.mark_attendance(student["id"], student["name"])
            if success:
                messagebox.showinfo("Success", f"Marked {student['name']} as Present")
                self.refresh_attendance_list()
                self.update_stats()
            else:
                messagebox.showinfo("Info", f"{student['name']}: {msg}")
            dialog.destroy()
        ttk.Button(dialog, text="Mark Present", command=mark, style="Accent.TButton").pack(pady=20)

    def show_records(self):
        self.set_active_menu("View Records")
        self.clear_main()
        self.stop_camera()
        ttk.Label(self.main_frame, text="Attendance Records", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(self.main_frame, text="View and export attendance history", style="Subtitle.TLabel").pack(anchor=tk.W, pady=(0, 20))
        filter_frame = ttk.Frame(self.main_frame, style="Dark.TFrame")
        filter_frame.pack(fill=tk.X, pady=10)
        ttk.Label(filter_frame, text="Date:", style="Dark.TLabel").pack(side=tk.LEFT, padx=(0, 5))
        self.filter_date = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(filter_frame, textvariable=self.filter_date, width=12, style="Dark.TEntry").pack(side=tk.LEFT, padx=(0, 15))
        ttk.Button(filter_frame, text="Filter", command=self.filter_records, style="Secondary.TButton").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(filter_frame, text="Show All", command=self.show_all_records, style="Secondary.TButton").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(filter_frame, text="Export CSV", command=self.export_csv, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(filter_frame, text="Export Excel", command=self.export_excel, style="Accent.TButton").pack(side=tk.LEFT)
        tree_frame = ttk.Frame(self.main_frame, style="Card.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        columns = ("Date", "Time", "StudentID", "Name", "Status")
        self.records_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        for col in columns:
            self.records_tree.heading(col, text=col)
            self.records_tree.column(col, width=150)
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.records_tree.yview)
        self.records_tree.configure(yscrollcommand=scrollbar.set)
        self.records_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        self.show_all_records()

    def filter_records(self):
        date = self.filter_date.get()
        df = self.attendance_mgr.get_all_attendance()
        if df.empty:
            return
        filtered = df[df["Date"] == date]
        self.populate_records_tree(filtered)

    def show_all_records(self):
        df = self.attendance_mgr.get_all_attendance()
        self.populate_records_tree(df)

    def populate_records_tree(self, df):
        for item in self.records_tree.get_children():
            self.records_tree.delete(item)
        if df.empty:
            return
        for _, row in df.iterrows():
            self.records_tree.insert("", tk.END, values=(row["Date"], row["Time"], row["StudentID"], row["Name"], row["Status"]))

    def export_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if path:
            df = self.attendance_mgr.get_all_attendance()
            df.to_csv(path, index=False)
            messagebox.showinfo("Success", f"Exported to {path}")

    def export_excel(self):
        try:
            path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
            if path:
                df = self.attendance_mgr.get_all_attendance()
                df.to_excel(path, index=False)
                messagebox.showinfo("Success", f"Exported to {path}")
        except ImportError:
            messagebox.showerror("Error", "Please install openpyxl: pip install openpyxl")

    def show_students(self):
        self.set_active_menu("Students")
        self.clear_main()
        self.stop_camera()
        ttk.Label(self.main_frame, text="Student Management", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(self.main_frame, text="View and manage registered students", style="Subtitle.TLabel").pack(anchor=tk.W, pady=(0, 20))
        btn_frame = ttk.Frame(self.main_frame, style="Dark.TFrame")
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="Refresh", command=self.refresh_students, style="Secondary.TButton").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Delete Selected", command=self.delete_student, style="Danger.TButton").pack(side=tk.LEFT)
        tree_frame = ttk.Frame(self.main_frame, style="Card.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        columns = ("ID", "Name", "Roll", "Email", "Registered", "Images")
        self.students_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        for col in columns:
            self.students_tree.heading(col, text=col)
            self.students_tree.column(col, width=150)
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.students_tree.yview)
        self.students_tree.configure(yscrollcommand=scrollbar.set)
        self.students_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        self.refresh_students()

    def refresh_students(self):
        for item in self.students_tree.get_children():
            self.students_tree.delete(item)
        students = load_students()
        for s in students:
            self.students_tree.insert("", tk.END, values=(s["id"], s["name"], s.get("roll", ""), s.get("email", ""), s.get("registered", ""), s.get("images", 0)))

    def delete_student(self):
        sel = self.students_tree.selection()
        if not sel:
            messagebox.showwarning("Warning", "Please select a student to delete")
            return
        item = self.students_tree.item(sel[0])
        student_id = item["values"][0]
        name = item["values"][1]
        if not messagebox.askyesno("Confirm", f"Delete student {name} ({student_id})?\nThis will remove all their face data."):
            return
        students = load_students()
        students = [s for s in students if s["id"] != student_id]
        save_students(students)
        import shutil
        student_dir = os.path.join(CONFIG["dataset_dir"], student_id)
        if os.path.exists(student_dir):
            shutil.rmtree(student_dir)
        self.face_engine.trained = False
        self.model_status.config(text="Model: Not Trained", foreground="#f38ba8")
        self.refresh_students()
        self.update_stats()
        messagebox.showinfo("Success", f"Student {name} deleted successfully")

    def show_train(self):
        self.set_active_menu("Train Model")
        self.clear_main()
        self.stop_camera()
        ttk.Label(self.main_frame, text="Train Recognition Model", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(self.main_frame, text="Train the KNN model on all registered faces", style="Subtitle.TLabel").pack(anchor=tk.W, pady=(0, 20))
        content = ttk.Frame(self.main_frame, style="Dark.TFrame")
        content.pack(fill=tk.BOTH, expand=True)
        info = ttk.Frame(content, style="Card.TFrame", padding=20)
        info.pack(fill=tk.X, pady=10)
        students = load_students()
        ttk.Label(info, text=f"Registered Students: {len(students)}", style="Card.TLabel", font=("Segoe UI", 12)).pack(anchor=tk.W)
        ttk.Label(info, text=f"Images per student: {CONFIG['captures_per_student']}", style="Card.TLabel").pack(anchor=tk.W, pady=5)
        ttk.Label(info, text=f"KNN Neighbors: {CONFIG['knn_neighbors']}", style="Card.TLabel").pack(anchor=tk.W)
        ttk.Label(info, text=f"Algorithm: Ball Tree with Euclidean distance", style="Card.TLabel").pack(anchor=tk.W, pady=5)
        self.train_progress = ttk.Progressbar(content, mode="determinate", maximum=100, style="Dark.Horizontal.TProgressbar")
        self.train_progress.pack(fill=tk.X, pady=20)
        self.train_status = ttk.Label(content, text="Ready to train", style="Dark.TLabel", font=("Segoe UI", 12))
        self.train_status.pack(anchor=tk.W)
        self.train_log = tk.Text(content, height=15, bg="#2d2d44", fg="#cdd6f4", font=("Consolas", 10), bd=0, insertbackground="#cdd6f4")
        self.train_log.pack(fill=tk.BOTH, expand=True, pady=10)
        scrollbar = ttk.Scrollbar(self.train_log, orient=tk.VERTICAL, command=self.train_log.yview)
        self.train_log.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        btn_frame = ttk.Frame(content, style="Dark.TFrame")
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="Start Training", command=self.start_training, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Clear Log", command=lambda: self.train_log.delete(1.0, tk.END), style="Secondary.TButton").pack(side=tk.LEFT)

    def start_training(self):
        students = load_students()
        if not students:
            messagebox.showerror("Error", "No students registered. Please register students first.")
            return
        self.train_log.delete(1.0, tk.END)
        self.train_log.insert(tk.END, "Starting training process...\n")
        self.train_log.insert(tk.END, f"Found {len(students)} students\n")
        self.train_log.insert(tk.END, "-" * 50 + "\n")
        def progress_callback(current, total, name, enc_count):
            pct = (current / total) * 100
            self.train_progress["value"] = pct
            self.train_status.config(text=f"Processing {name}... ({current}/{total})")
            self.train_log.insert(tk.END, f"[{current}/{total}] {name}: {enc_count} encodings\n")
            self.train_log.see(tk.END)
            self.root.update_idletasks()
        def train_thread():
            success, msg = self.face_engine.train_model(progress_callback)
            self.root.after(0, self.training_complete, success, msg)
        threading.Thread(target=train_thread, daemon=True).start()

    def training_complete(self, success, msg):
        self.train_progress["value"] = 100 if success else 0
        if success:
            self.train_status.config(text="Training complete!", foreground="#a6e3a1")
            self.train_log.insert(tk.END, "\n" + "=" * 50 + "\n")
            self.train_log.insert(tk.END, f"SUCCESS: {msg}\n")
            self.model_status.config(text="Model: Trained", foreground="#a6e3a1")
            messagebox.showinfo("Training Complete", msg)
        else:
            self.train_status.config(text=f"Training failed: {msg}", foreground="#f38ba8")
            self.train_log.insert(tk.END, f"\nERROR: {msg}\n")
            messagebox.showerror("Training Failed", msg)

    def show_settings(self):
        self.set_active_menu("Settings")
        self.clear_main()
        self.stop_camera()
        ttk.Label(self.main_frame, text="Settings", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(self.main_frame, text="Configure system parameters", style="Subtitle.TLabel").pack(anchor=tk.W, pady=(0, 20))
        form = ttk.Frame(self.main_frame, style="Card.TFrame", padding=20)
        form.pack(fill=tk.X, pady=10)
        ttk.Label(form, text="KNN Neighbors", style="Card.TLabel").pack(anchor=tk.W)
        self.knn_var = tk.IntVar(value=CONFIG["knn_neighbors"])
        ttk.Spinbox(form, from_=1, to=10, textvariable=self.knn_var, width=10).pack(anchor=tk.W, pady=(0, 10))
        ttk.Label(form, text="Recognition Tolerance", style="Card.TLabel").pack(anchor=tk.W)
        self.tol_var = tk.DoubleVar(value=CONFIG["tolerance"])
        ttk.Spinbox(form, from_=0.1, to=1.0, increment=0.1, textvariable=self.tol_var, width=10).pack(anchor=tk.W, pady=(0, 10))
        ttk.Label(form, text="Captures per Student", style="Card.TLabel").pack(anchor=tk.W)
        self.caps_var = tk.IntVar(value=CONFIG["captures_per_student"])
        ttk.Spinbox(form, from_=5, to=20, textvariable=self.caps_var, width=10).pack(anchor=tk.W, pady=(0, 10))
        ttk.Label(form, text="Camera Index", style="Card.TLabel").pack(anchor=tk.W)
        self.cam_var = tk.IntVar(value=CONFIG["camera_index"])
        ttk.Spinbox(form, from_=0, to=5, textvariable=self.cam_var, width=10).pack(anchor=tk.W, pady=(0, 10))
        ttk.Button(form, text="Save Settings", command=self.save_settings, style="Accent.TButton").pack(anchor=tk.W, pady=(20, 0))
        ttk.Label(self.main_frame, text="Data Management", style="Dark.TLabel", font=("Segoe UI", 14, "bold")).pack(anchor=tk.W, pady=(30, 10))
        data_frame = ttk.Frame(self.main_frame, style="Card.TFrame", padding=20)
        data_frame.pack(fill=tk.X, pady=10)
        ttk.Button(data_frame, text="Clear All Attendance Records", command=self.clear_attendance, style="Danger.TButton").pack(anchor=tk.W, pady=(0, 5))
        ttk.Button(data_frame, text="Reset All Data", command=self.reset_all_data, style="Danger.TButton").pack(anchor=tk.W)

    def save_settings(self):
        CONFIG["knn_neighbors"] = self.knn_var.get()
        CONFIG["tolerance"] = self.tol_var.get()
        CONFIG["captures_per_student"] = self.caps_var.get()
        CONFIG["camera_index"] = self.cam_var.get()
        messagebox.showinfo("Success", "Settings saved successfully!")

    def clear_attendance(self):
        if messagebox.askyesno("Confirm", "Delete all attendance records? This cannot be undone."):
            if os.path.exists(CONFIG["attendance_file"]):
                os.remove(CONFIG["attendance_file"])
            self.attendance_mgr.attendance_cache = {}
            messagebox.showinfo("Success", "All attendance records cleared")

    def reset_all_data(self):
        if messagebox.askyesno("Confirm", "Reset ALL data? This will delete students, attendance, and trained model."):
            import shutil
            for path in [CONFIG["dataset_dir"], CONFIG["attendance_file"], CONFIG["model_file"], CONFIG["students_file"]]:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
            self.face_engine.trained = False
            self.model_status.config(text="Model: Not Trained", foreground="#f38ba8")
            messagebox.showinfo("Success", "All data reset. Please restart the application.")

    def stop_camera(self):
        self.auto_recognize = False
        if self.camera:
            self.camera.stop()
            self.camera = None
        self.set_status("Camera stopped")

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = SmartAttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
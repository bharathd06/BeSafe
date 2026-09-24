# AI-Powered Multi-Threat Video Surveillance & Crime Detection System

An intelligent real-time surveillance pipeline using **YOLO (You Only Look Once)** deep learning object detection models combined with **DeepSORT** multi-object tracking to detect security threats in CCTV/video feeds and categorize them by severity.

---

## 🚀 Key Features

- **🔫 Weapon Detection**: Real-time detection of handguns and firearms.
- **🪖 Helmet Detection**: Detects helmets worn in suspicious, non-traffic environments (e.g., banks, stores).
- **🎭 Face Cover & Mask Detection**: Identifies ski-masks and covered faces attempting identity concealment.
- **🏃 Human Movement & Rapid Action Tracking**: DeepSORT tracking calculates person velocity across frames to distinguish normal walking from rapid/fleeing movement (running).
- **⭐ 5-Tier Threat Rating System**: Automatically classifies detected activities into 1 to 5 star severity levels.
- **📸 Visual HUD & Snapshot Archiving**: Overlays bounding boxes and threat banners in real-time, saving all flagged suspicious frames with timestamps.

---

## ⭐ Threat Severity Rating System

| Rating | Threat Level | Trigger Condition | Visual Indicator |
| :---: | :---: | :--- | :--- |
| ⭐⭐⭐⭐⭐ | **Level 5: Critical** | Firearm / Gun detected | Red Box & Alert |
| ⭐⭐⭐⭐ | **Level 4: High** | Ski-mask (concealed face) or Helmet detected | Orange Box & Alert |
| ⭐⭐⭐ | **Level 3: Moderate** | Prolonged running / fleeing movement ($\ge 20$ frames) | Red Box & Running Tag |
| ⭐⭐ | **Level 2: Caution** | Short burst of running movement ($\ge 10$ frames) | Red Box & Running Tag |
| ⭐ | **Level 1: Normal** | Standard pedestrian walking / no threats | Green Box & Normal Tag |

---

## 📁 Repository Structure

`	ext
├── EL/
│   └── final/                 # Trained YOLO weights (.pt) & training checkpoints
│       ├── guns/              # Gun detection model weights
│       ├── helmets/           # Helmet detection model weights
│       ├── ski mask/          # Face cover / ski-mask weights
│       └── yolo11n.pt         # YOLOv11 person tracking model
├── python files/
│   └── finalel.py             # Main comprehensive detection & threat rating script
├── crime_server/              # Edge computing / Raspberry Pi server (Flask backend)
│   ├── crime.py               # Server-side detection worker
│   └── pi_server.py           # REST API endpoint for video processing
├── web_app/                   # Client Flask web application
│   ├── app.py                 # Web app routing & upload handling
│   ├── templates/             # HTML templates (upload & results display)
│   └── static/                # Static assets & styles
├── test images/               # Sample test frames for benchmarking
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
`

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
`ash
git clone <YOUR_REPOSITORY_URL>
cd <REPOSITORY_FOLDER>
`

### 2. Create and Activate a Virtual Environment
`ash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
`

### 3. Install Dependencies
`ash
pip install -r requirements.txt
`

---

## 💻 Running the Detection Pipeline

### Run on a Video File:
`ash
python "python files/finalel.py" "path/to/your_video.mp4"
`

### Run on Live Webcam Feed:
`ash
python "python files/finalel.py" 0
`

- Annotated video is saved automatically to python files/output_combined.mp4.
- Suspicious detection snapshots are saved with timestamps to python files/suspicious_frames/.
- Press **q** in the video display window to stop processing.

---

## 👥 Authors & Academic Details

- **Bharath D** (1RV23CS062)
- **Bhavan TG** (1RV23CS063)
- **Girish Goudar** (1RV24CS405)
- Department of Computer Science & Engineering, RV College of Engineering (RVCE)

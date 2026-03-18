# 🎖️ Real-Time Multi-Camera Emotion Detection System

> **Project 2 — Indian Army 17th Zonal CORP**  
> An advanced surveillance and behavioral analysis system capable of detecting and monitoring human emotions across multiple camera feeds simultaneously in real time.

---

## 📋 Table of Contents

- [About](#about)
- [Key Features](#key-features)
- [Detected Emotions](#detected-emotions)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the System](#running-the-system)
- [Configuration](#configuration)
- [Use Cases](#use-cases)
- [Security & Ethics](#security--ethics)
- [Related Projects](#related-projects)
- [License](#license)

---

## About

The **Real-Time Multi-Camera Emotion Detection System** is a defense-grade AI application developed for the Indian Army Corps. It processes live video streams from multiple cameras simultaneously, detecting human faces and classifying their emotional states in real time.

Built on top of deep learning models and computer vision pipelines, the system is designed for high-throughput, low-latency deployment in controlled environments such as checkpoints, security zones, and monitored perimeters.

This is the second project in a series of AI-powered security tools developed for the Army Corps, following [ArmyFaceDetection](https://github.com/PalashJyoti/ArmyFaceDetection).

---

## Key Features

- 🎥 **Multi-Camera Support** — Simultaneously processes feeds from multiple cameras (USB, IP, or RTSP streams)
- ⚡ **Real-Time Processing** — Low-latency emotion inference per frame using optimized deep learning models
- 🧠 **7-Class Emotion Recognition** — Classifies all major human emotional states
- 📦 **Full-Stack Architecture** — Python backend for AI processing + JavaScript frontend dashboard
- 📊 **Live Dashboard** — Visual display of emotion data per camera feed with bounding boxes and labels
- 🔔 **Alert System** — Configurable alerts triggered by specific emotions (e.g., anger, fear)
- 💾 **Logging & History** — Emotion events are timestamped and stored for post-analysis
- 🔒 **Secured for Deployment** — Designed for closed-network, on-premise military environments

---

## Detected Emotions

The system can classify the following emotional states from facial expressions:

| Emotion   | Description                                      |
|-----------|--------------------------------------------------|
| 😠 Angry   | Signs of aggression or hostility                |
| 🤢 Disgust | Expressions of aversion or contempt             |
| 😨 Fear    | Indicators of threat perception or distress     |
| 😊 Happy   | Positive emotional states                       |
| 😢 Sad     | Signs of grief or low morale                    |
| 😲 Surprise| Unexpected or alert states                      |
| 😐 Neutral | Calm, baseline emotional state                  |

---

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Camera Input Layer                  │
│     Camera 1 │ Camera 2 │ Camera 3 │ ... Camera N   │
└──────────────────────┬───────────────────────────────┘
                       │ Video Frames
                       ▼
┌──────────────────────────────────────────────────────┐
│               Python Backend (AI Engine)             │
│                                                      │
│  ┌────────────────┐     ┌─────────────────────────┐  │
│  │  Face Detection │────▶│  Emotion Classification │  │
│  │  (OpenCV /      │     │  (DeepFace / FER / CNN) │  │
│  │  Haar Cascade)  │     └──────────┬──────────────┘  │
│  └────────────────┘                │                  │
│                                    ▼                  │
│                         ┌──────────────────┐          │
│                         │  Alert Engine &  │          │
│                         │  Event Logger    │          │
│                         └──────────────────┘          │
└───────────────────────────┬──────────────────────────┘
                            │ WebSocket / REST API
                            ▼
┌──────────────────────────────────────────────────────┐
│              JavaScript Frontend Dashboard           │
│   Live camera feeds │ Emotion overlays │ Alert panel │
└──────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer            | Technology                                          |
|------------------|-----------------------------------------------------|
| Backend Language | Python 3.8+                                         |
| Computer Vision  | OpenCV (`cv2`)                                      |
| Emotion AI       | DeepFace / FER / Custom CNN (TensorFlow + Keras)    |
| Face Detection   | Haar Cascade / MTCNN                                |
| API Server       | Flask / FastAPI                                     |
| Real-Time Comm   | WebSockets (Flask-SocketIO)                         |
| Frontend         | JavaScript (React / Vanilla JS)                     |
| Dataset          | FER-2013 (35,000+ labeled facial images)            |
| Deployment       | On-premise / Closed-network                         |

---

## Project Structure

```
RealTimeEmotionDetection/
├── backend/
│   ├── app.py                  # Main server & WebSocket handler
│   ├── emotion_detector.py     # Core emotion detection logic
│   ├── camera_manager.py       # Multi-camera stream management
│   ├── alert_engine.py         # Configurable alert system
│   ├── logger.py               # Emotion event logging
│   ├── models/                 # Pre-trained model weights
│   │   └── emotion_model.h5
│   ├── utils/
│   │   └── face_utils.py       # Face ROI extraction helpers
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── CameraFeed.js   # Individual camera feed component
│   │   │   ├── Dashboard.js    # Multi-feed dashboard layout
│   │   │   └── AlertPanel.js   # Real-time alert display
│   │   └── App.js
│   ├── public/
│   └── package.json
│
└── README.md
```

---

## Getting Started

### Prerequisites

- Python **3.8+**
- Node.js **16+** and npm
- One or more connected cameras (USB webcams, IP cameras, or RTSP streams)
- GPU recommended for multi-camera real-time inference (CUDA-enabled)

---

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/PalashJyoti/RealTimeEmotionDetection.git
cd RealTimeEmotionDetection
```

**2. Set up the Python backend**

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Key dependencies:
```
opencv-python
deepface
tensorflow
keras
flask
flask-socketio
numpy
```

**3. Set up the JavaScript frontend**

```bash
cd ../frontend
npm install
```

---

### Running the System

**Start the backend server:**

```bash
cd backend
python app.py
```

Backend runs at: `http://localhost:5000`

**Start the frontend dashboard:**

```bash
cd frontend
npm start
```

Frontend runs at: `http://localhost:3000`

**Open your browser** and navigate to `http://localhost:3000` to view the live multi-camera emotion dashboard.

---

## Configuration

Edit `backend/config.py` to customize the system:

```python
# Camera sources (0 = default webcam, or RTSP URL for IP cameras)
CAMERAS = [
    0,                                      # USB Webcam 1
    1,                                      # USB Webcam 2
    "rtsp://192.168.1.10:554/stream",       # IP Camera
]

# Emotions that trigger an alert
ALERT_EMOTIONS = ["angry", "fear"]

# Detection confidence threshold (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.65

# Frames per second target
TARGET_FPS = 15

# Enable/disable event logging
ENABLE_LOGGING = True
LOG_PATH = "logs/emotion_events.csv"
```

---

## Use Cases

- **Checkpoint Monitoring** — Automated behavioral screening at entry/exit points
- **Perimeter Surveillance** — Passive emotional state monitoring in sensitive zones
- **Personnel Welfare** — Early detection of stress, fear, or distress in personnel
- **Crowd Analysis** — Aggregated emotional state assessment during gatherings or operations
- **Post-Incident Review** — Replay and analysis of logged emotion events

---

## Security & Ethics

This system was developed exclusively for authorized military and defense use. Deployment must comply with:

- ✅ Authorized personnel monitoring only within secured, closed-network environments
- ✅ Data is processed and stored entirely on-premise — no cloud transmission
- ✅ Access restricted to authorized operators with appropriate clearance
- ⚠️ Emotion recognition AI carries inherent accuracy limitations — human review is required for critical decisions
- ⚠️ Not intended for use on civilian populations without explicit legal authorization

---

## Related Projects

| Project | Description |
|---------|-------------|
| [ArmyFaceDetection](https://github.com/PalashJyoti/ArmyFaceDetection) | Project 1 — Face detection and identification system for Army Corps |
| Real-Time Multi-Camera Emotion Detection | **Project 2 — This repository** |

---

## License

Developed for and in partnership with the **Indian Army Corps**. All rights reserved. Unauthorized distribution or deployment outside of authorized environments is strictly prohibited.

---

> Built by [PalashJyoti](https://github.com/PalashJyoti) · Indian Army Corps AI Initiative

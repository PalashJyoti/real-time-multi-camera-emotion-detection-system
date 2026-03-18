# 🎖️ Real-Time Multi-Camera Emotion Detection System

> **Project 2 — Indian Army 17th Zonal Corps**  
> A defense-grade full-stack system that detects and monitors human emotions across multiple live camera feeds simultaneously using deep learning and computer vision.

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
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Pages Overview](#pages-overview)
- [Use Cases](#use-cases)
- [Security & Ethics](#security--ethics)
- [Related Projects](#related-projects)
- [License](#license)

---

## About

The **Real-Time Multi-Camera Emotion Detection System** is a defense-grade AI application developed for the Indian Army Corps. It processes live video streams from multiple cameras simultaneously, detecting human faces and classifying their emotional states in real time.

The backend is built with **Python + Flask**, using a **Caffe-based SSD ResNet-10 face detector** paired with a custom **emotion classification CNN**. Multi-threading handles concurrent camera streams with dedicated `FrameCaptureThread` and `EmotionDetectorThread` instances per camera, all orchestrated by a `MultiCameraManager`.

The frontend is a **Next.js** web application featuring a live multi-camera dashboard, interactive emotion pie charts, webcam support, video file analysis, full user authentication, and an admin panel.

This is the second project in a series of AI-powered security tools developed for the Army Corps, following [ArmyFaceDetection](https://github.com/PalashJyoti/ArmyFaceDetection).

---

## Key Features

- 🎥 **Multi-Camera Support** — Manages multiple simultaneous camera streams via `MultiCameraManager`
- ⚡ **Threaded Real-Time Processing** — Dedicated `FrameCaptureThread` and `EmotionDetectorThread` per camera for non-blocking parallel inference
- 🧠 **Deep Learning Pipeline** — Caffe SSD ResNet-10 for face detection + custom CNN for 7-class emotion classification
- 📊 **Interactive Dashboard** — Live per-camera emotion overlays with an interactive pie chart (`interactivePieChart.js`)
- 🔐 **Full Authentication System** — Signup, login, and forgot password flows with route-level guards
- 👤 **Admin Panel** — Admin controls for managing cameras and system configuration
- 📹 **Webcam & Video Support** — Live webcam emotion detection and video file emotion analysis pages
- 📜 **Event Logs** — View and review historical emotion detection event logs
- 📡 **Resource Monitoring** — Backend tracks CPU/memory usage in real time via `ResourceMonitor`
- 🔒 **On-Premise Deployment** — Designed for closed-network, secured military environments

---

## Detected Emotions

| Emotion    | Description                                  |
|------------|----------------------------------------------|
| 😠 Angry   | Signs of aggression or hostility             |
| 🤢 Disgust | Expressions of aversion or contempt         |
| 😨 Fear    | Indicators of distress or threat perception |
| 😊 Happy   | Positive or relaxed emotional state         |
| 😢 Sad     | Signs of grief or low morale                |
| 😲 Surprise| Alert or unexpected reaction                |
| 😐 Neutral | Calm, baseline emotional state              |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Camera Input Layer                      │
│          Camera 1 │ Camera 2 │ Camera 3 │ ... Camera N      │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│          multi_camera_manager.py  (MultiCameraManager)      │
│                                                             │
│  Per Camera:                                                │
│  ┌─────────────────────┐     ┌──────────────────────────┐  │
│  │  FrameCaptureThread │────▶│  EmotionDetectorThread   │  │
│  │  frame_capture_     │     │  emotion_detector_       │  │
│  │  thread.py          │     │  thread.py               │  │
│  └─────────────────────┘     └────────────┬─────────────┘  │
│                                           │                │
│                 ┌─────────────────────────▼─────────────┐  │
│                 │  Face Detection                        │  │
│                 │  res10_300x300_ssd_iter_140000         │  │
│                 │  .caffemodel  +  deploy.prototxt       │  │
│                 └─────────────────┬───────────────────── ┘  │
│                                   │                        │
│                 ┌─────────────────▼─────────────────────┐  │
│                 │  Emotion Classification  (model.py)    │  │
│                 │  predict.py                            │  │
│                 └─────────────────┬─────────────────────┘  │
│                                   │                        │
│           ┌───────────────────────▼──────────────────┐     │
│           │  logger.py  │  globals.py  │  resource_  │     │
│           │             │              │  monitor.py │     │
│           └─────────────────────────────────────────-┘     │
└─────────────────────────┬───────────────────────────────────┘
                          │  REST API (Flask)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Next.js Frontend                         │
│  /dashboard  │  /emotion  │  /videoemotion  │  /logs        │
│  /admin      │  /login    │  /signup        │  /forgot      │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer              | Technology                                                    |
|--------------------|---------------------------------------------------------------|
| Backend Language   | Python 3.8+                                                   |
| Web Framework      | Flask                                                         |
| Face Detection     | OpenCV + Caffe SSD ResNet-10 (`res10_300x300_ssd_iter_140000`) |
| Emotion Model      | Custom CNN — `model.py` + `predict.py`                        |
| Concurrency        | Python threading (`FrameCaptureThread`, `EmotionDetectorThread`) |
| DB / Models        | SQLAlchemy via `extensions.py` + `models.py`                  |
| Frontend Framework | Next.js (React)                                               |
| HTTP Client        | Axios (`pages/api/axios.js`)                                  |
| Styling            | Global CSS + PostCSS                                          |
| Charts             | Custom `InteractivePieChart` component                        |

---

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── auth/
│   │   │   ├── __init__.py
│   │   │   └── routes.py               # Auth API routes (login, signup, forgot password)
│   │   ├── camera/
│   │   │   ├── camera_manager.py       # App-level camera management
│   │   │   ├── emotion_worker.py       # Emotion processing worker
│   │   │   ├── model.py                # Camera DB model
│   │   │   └── routes.py               # Camera API routes
│   │   └── __init__.py                 # Flask app factory
│   │
│   ├── emotion_detection_service/
│   │   ├── __init__.py
│   │   ├── deploy.prototxt             # Caffe SSD face detector network config
│   │   ├── emotion_detector_thread.py  # Per-camera emotion detection thread
│   │   ├── emotion_worker_service.py   # Emotion worker orchestration
│   │   ├── exception.py                # Service-level custom exceptions
│   │   ├── frame_capture_thread.py     # Per-camera frame capture thread
│   │   ├── globals.py                  # Shared global state across threads
│   │   ├── logger.py                   # Detection service logger
│   │   ├── model.py                    # Emotion model loader & inference
│   │   ├── multi_camera_manager.py     # Orchestrates all camera threads
│   │   ├── predict.py                  # Emotion prediction pipeline
│   │   ├── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained face detector weights
│   │   ├── resource_monitor.py         # CPU/memory resource monitoring
│   │   └── routes.py                   # Detection service API routes
│   │
│   ├── exception.py                    # Global exception handlers
│   ├── extensions.py                   # Flask extensions (db, jwt, etc.)
│   ├── ip.py                           # IP/network utilities
│   ├── logger.py                       # App-level logger
│   ├── models.py                       # Shared SQLAlchemy database models
│   ├── requirements.txt
│   └── run.py                          # Application entry point
│
└── frontend/
    ├── components/
    │   ├── authNav.js                  # Authenticated navigation bar
    │   ├── cameraRow.js                # Per-camera feed row component
    │   ├── card.js                     # Reusable card UI component
    │   ├── interactivePieChart.js      # Real-time emotion distribution pie chart
    │   ├── navbar.js                   # Main navigation bar
    │   ├── spinner.js                  # Loading spinner
    │   └── webcamComponent.js          # Live webcam capture component
    │
    ├── pages/
    │   ├── api/
    │   │   ├── axios.js                # Axios instance with base URL & interceptors
    │   │   └── hello.js                # Next.js API health check route
    │   ├── _app.js                     # Global app wrapper
    │   ├── _document.js                # Custom HTML document
    │   ├── admin.js                    # Admin panel
    │   ├── dashboard.js                # Live multi-camera emotion dashboard
    │   ├── emotion.js                  # Emotion detection results & analytics
    │   ├── forgot.js                   # Forgot password page
    │   ├── index.js                    # Landing / home page
    │   ├── login.js                    # Login page
    │   ├── logs.js                     # Emotion event logs viewer
    │   ├── signup.js                   # User registration page
    │   └── videoemotion.js             # Video file emotion analysis
    │
    ├── styles/
    │   └── globals.css
    ├── public/
    ├── .env.local                      # Environment variables (API base URL, etc.)
    ├── next.config.mjs
    └── package.json
```

---

## Getting Started

### Prerequisites

- Python **3.8+**
- Node.js **18+** and npm
- One or more cameras (USB webcams, IP cameras, or RTSP streams)
- GPU recommended for multi-camera real-time inference

---

### Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python run.py
```

Backend runs at: `http://localhost:5000`

---

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment variables
# Edit .env.local and set your backend URL:
# NEXT_PUBLIC_API_URL=http://localhost:5000

# Start the development server
npm run dev
```

Frontend runs at: `http://localhost:3000`

---

## Pages Overview

| Route            | File                | Description                                     |
|------------------|---------------------|-------------------------------------------------|
| `/`              | `index.js`          | Landing / home page                             |
| `/login`         | `login.js`          | User login                                      |
| `/signup`        | `signup.js`         | New user registration                           |
| `/forgot`        | `forgot.js`         | Password recovery                               |
| `/dashboard`     | `dashboard.js`      | Live multi-camera emotion detection dashboard   |
| `/emotion`       | `emotion.js`        | Emotion detection results and analytics         |
| `/videoemotion`  | `videoemotion.js`   | Emotion analysis on uploaded video files        |
| `/logs`          | `logs.js`           | Historical emotion detection event logs         |
| `/admin`         | `admin.js`          | Admin panel for system and camera management    |

---

## Use Cases

- **Checkpoint Monitoring** — Automated behavioral screening at entry/exit points
- **Perimeter Surveillance** — Passive emotional state monitoring in sensitive zones
- **Personnel Welfare** — Early detection of stress, fear, or distress among personnel
- **Video Review** — Post-incident emotion analysis on recorded footage via `/videoemotion`
- **Operational Logging** — Full audit trail of emotion events via the `/logs` page

---

## Security & Ethics

This system was developed exclusively for authorized military and defense use:

- ✅ All data is processed and stored entirely **on-premise** — no cloud transmission
- ✅ Access is gated behind a full authentication system with admin controls
- ✅ Camera management is restricted to authorized operators via the admin panel
- ⚠️ Emotion recognition AI has inherent accuracy limitations — human review is required for all critical decisions
- ⚠️ Not intended for deployment on civilian populations without explicit legal authorization

---

## Related Projects

| # | Project | Description |
|---|---------|-------------|
| 1 | [ArmyFaceDetection](https://github.com/PalashJyoti/ArmyFaceDetection) | Face detection and identification system for Army Corps |
| 2 | Real-Time Multi-Camera Emotion Detection | **This project** — Multi-camera real-time emotion monitoring |

---

## License

Developed for and in partnership with the **Indian Army Corps**. All rights reserved.  
Unauthorized distribution or deployment outside authorized environments is strictly prohibited.

---

> Built by [PalashJyoti](https://github.com/PalashJyoti) · Indian Army Corps AI Initiative

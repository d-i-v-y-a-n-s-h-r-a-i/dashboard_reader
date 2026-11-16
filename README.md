# TANDAV Dashboard Reader: Optical Telemetry for Autonomous Boats

> **Real-Time Optical Telemetry Extraction Engine** designed for autonomous marine vessels operating in pitch-black or low-light cabin environments, extracting numerical dashboard instrumentation from an illuminated orange instrument cluster via a camera and streaming high-frequency debounced telemetry to operator ground-station software.

---

## 🚤 Problem & Solution Overview

In autonomous surface vessels (ASVs) or remote-operated boats, physical CAN/NMEA dashboard wires may be inaccessible, void warranties, or lack digital outputs. The cockpit dashboard has an orange backlight LCD displaying speed, mode, and fuel. 

This system:
1. Captures live video from an internal camera (global shutter camera on Jetson Orin or USB webcam).
2. Isolates the orange-backlight LCD using adaptive HSV color bounds.
3. Automatically computes 4-corner perspective rectification with Exponential Moving Average (EMA) tremor-filtering to eliminate camera vibration.
4. Uses topological 7-segment OCR to classify digits (0–99+ km/h) dynamically without fixed-slot misalignment.
5. Applies temporal consensus debouncing to discard optical glitches, glare, or transient occlusion.
6. Streams validated JSON telemetry packets over UDP (port `47001`) to the operator's ground-control station UI.

---

## 💻 Hardware Requirements

- **Processor**: NVIDIA Jetson Orin (Nano / NX / AGX), Intel/AMD Mini-PC, or any modern Windows/Linux laptop.
- **Camera**:
  - *Production*: Global Shutter USB or MIPI-CSI camera (e.g. OV9281, Arducam, or Sony IMX sensor) pointed directly at the boat cluster.
  - *Benchtop Testing*: Any standard laptop webcam or USB webcam pointing at a phone screen playing the cluster video.

---

## 📦 What to Download on a New Laptop

To run this project on a clean laptop, you only need:
1. **Python 3.10+** (Python 3.10, 3.11, 3.12, 3.13, or 3.14).
   - [Download Python from python.org](https://www.python.org/downloads/) (Make sure to check *"Add python.exe to PATH"* during Windows installation).
2. **Git** (to clone the repo).
   - [Download Git for Windows/Linux/Mac](https://git-scm.com/downloads).

All other Python libraries (`numpy`, `opencv-python`, `PyYAML`, `aiohttp`) are installed automatically via `pip`.

---

## 🚀 Setup Instructions (Step-by-Step)

### 🪟 Windows (PowerShell)

1. **Clone the Repository**:
   ```powershell
   git clone https://github.com/d-i-v-y-a-n-s-h-r-a-i/dashboard_reader.git
   cd dashboard_reader
   ```

2. **Create a Virtual Environment**:
   ```powershell
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   *(If PowerShell displays a script execution policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once, then re-run the activation script).*

4. **Install Dependencies**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### 🐧 Linux / Jetson Orin / macOS (Bash)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/d-i-v-y-a-n-s-h-r-a-i/dashboard_reader.git
   cd dashboard_reader
   ```

2. **Create a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🧪 How to Run & Test

### Option 1: Direct Video Test (No Camera Required)
Run the pipeline against the included sample cluster video (`data/screen.mp4`):
```bash
python tools/test_video.py
```
- Opens a visualization window displaying the dashboard, slot detection, and instantaneous speed.
- Logs telemetry events to stdout and broadcasts over UDP.
- Press `Q` to quit, `SPACE` to pause/resume.

---

### Option 2: Live Webcam / Phone Testing (Interactive)
Test the live OCR engine with your laptop webcam while holding up a phone playing the dashboard video:
```bash
python tools/live_phone_test.py
```
- **Options**:
  - `--device 0`: Specify webcam index (default `0`).
  - `--udp-port 47001`: Custom UDP port.
- **Features**:
  - Automatically locks onto the orange phone screen.
  - Real-time perspective unwarping with smooth anti-jitter tracking.
  - HUD displays speed in km/h, confidence score, throttle, and detection FPS.

---

### Option 3: Sniff Live UDP Packets
In a second terminal window, verify that the operator telemetry packets are being broadcasted:
```bash
python tools/sniff_udp.py --port 47001
```

---

### Option 4: Production Pipeline & Validation Suite
Run the full autonomous boat pipeline:
```bash
python -m tandav run --config config/tandav.yaml
```

Run unit tests and self-test verification:
```bash
# Core unit tests
python -m unittest discover tests

# Synthetic benchmarking & gate validation (50+ FPS verification)
python -m tandav selftest --gate
```

---

## 📡 Telemetry Packet Protocol

The engine broadcasts telemetry datagrams over UDP (`127.0.0.1:47001`) in JSON format:

```json
{
  "page": "MAIN_TELEMETRY",
  "speed": 55.0,
  "rpm": null,
  "conf": 1.0,
  "valid": true,
  "t_mono_us": 1731778942000
}
```

### Field Definitions:
- `page`: Identified dashboard mode (`"MAIN_TELEMETRY"` or `"FUEL"`).
- `speed`: Numerical vessel speed extracted in km/h (`float` or `null` if occluded/unrecognized).
- `conf`: Confidence score between `0.0` and `1.0`.
- `valid`: Boolean flag indicating whether the frame meets consensus debouncing criteria.
- `t_mono_us`: Monotonic microsecond timestamp for latency compensation on the operator ground control.

---

## 📁 Repository Structure

```
dashboard_reader/
├── calibration/             # Calibrated LCD templates, anchor marks, and digit maps
│   ├── glyphs/             # Multi-font glyph references
│   ├── real_digits/        # Real 7-segment digit templates (0-9)
│   ├── anchor.png          # Visual anchor for cluster orientation
│   └── calibration.json    # Slot coordinate geometry
├── config/
│   └── tandav.yaml         # Camera, UDP, and pipeline configuration
├── data/
│   └── screen.mp4          # Recorded dashboard footage for offline validation
├── runtime/                # Directory for local execution logs and test captures
├── tandav/                 # Core production package
│   ├── align.py            # Homography alignment & perspective warping
│   ├── camera.py           # V4L2 / OpenCV asynchronous frame grabber
│   ├── kalman.py           # Kinematic velocity state filter
│   ├── pipeline.py         # End-to-end vision processing loop
│   ├── protocol.py         # Telemetry packet serializer
│   ├── sender.py           # UDP socket broadcaster
│   ├── selftest.py         # Latency and accuracy benchmarking suite
│   └── types.py            # Data classes and telemetry contracts
├── tests/
│   └── test_core.py        # Automated test suite
├── tools/
│   ├── live_phone_test.py  # Interactive live webcam + phone testing tool
│   ├── sniff_udp.py        # UDP receiver listener tool
│   └── test_video.py       # Offline video runner
├── requirements.txt        # Python dependency manifest
└── README.md               # Setup and usage guide
```

---

## 🛠️ Troubleshooting

- **Webcam is not detected (`Cannot open webcam index 0`)**:
  - Close any other applications using the webcam (Zoom, Teams, Camera app).
  - Pass `--device 1` or `--device 2` if using an external USB camera.
- **PowerShell Script Execution Restriction**:
  - Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell, then re-run `.\.venv\Scripts\Activate.ps1`.
- **OpenCV Display Error (`The function is not implemented`)**:
  - Ensure standard `opencv-python` is installed rather than `opencv-python-headless`.

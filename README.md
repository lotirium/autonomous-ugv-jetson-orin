# Rovy - AI Robot Assistant

Cloud-based robot assistant using Raspberry Pi + PC via Tailscale.

## Architecture

```
┌─────────────────────┐                              ┌─────────────────────┐
│     MOBILE APP      │◄────── REST API ───────────►│      PC CLOUD       │
│     (Phone)         │        (port 8000)           │   (This Machine)    │
│                     │                              │                     │
│ • Joystick control  │                              │ • FastAPI REST      │
│ • Camera view       │                              │ • LLM (Gemma/Llama) │
│ • Status display    │                              │ • Vision (LLaVA)    │
└─────────────────────┘                              │ • STT (Whisper)     │
                                                     │ • TTS (Piper)       │
                                                     │ • Face Recognition  │
                                                     └──────────┬──────────┘
                                                                │
                                                           WebSocket
                                                          (port 8765)
                                                                │
                                                     ┌──────────▼──────────┐
                                                     │   RASPBERRY PI      │
                                                     │   (On Robot)        │
                                                     │                     │
                                                     │ • Rover control     │
                                                     │ • Camera streaming  │
                                                     │ • Microphone        │
                                                     │ • Speaker           │
                                                     └─────────────────────┘
```

## Folder Structure

```
rovy/
├── cloud/              # 👈 Runs on PC (this machine)
│   ├── main.py         # Unified server (REST API + WebSocket)
│   ├── ai.py           # LLM/Vision models (llama.cpp)
│   ├── speech.py       # STT (Whisper) + TTS (Piper)
│   ├── app/            # FastAPI REST endpoints
│   └── config.py       # Server configuration
│
├── robot/              # 👈 Runs on Raspberry Pi
│   ├── main.py         # Client connecting to cloud
│   ├── rover.py        # Serial control for ESP32
│   └── config.py       # Robot configuration
│
├── mobile/             # 👈 React Native app (Expo)
│   ├── app/            # Screens
│   ├── components/     # UI components
│   └── services/       # API clients
│
├── firmware/           # ESP32 rover firmware
│   └── ugv_base_general/
│
└── archive/            # Old code
    └── jetson_legacy/  # Original Jetson code
```

## Quick Start

### 1. On PC (Cloud Server)

```bash
cd cloud
pip install -r requirements.txt
python main.py
```

This starts:
- **REST API** on `http://0.0.0.0:8000` (for mobile app)
- **WebSocket** on `ws://0.0.0.0:8765` (for robot)

### 2. On Raspberry Pi (Robot)

```bash
cd robot
pip install -r requirements.txt

# Set your PC's Tailscale IP
export ROVY_PC_IP=100.121.110.125

python main.py
```

### 3. Mobile App

```bash
cd mobile
npm install
npx expo start
```

## Network (Tailscale)

| Device       | Tailscale IP    | Ports              |
|--------------|-----------------|-------------------|
| PC           | 100.121.110.125 | 8000 (REST), 8765 (WS) |
| Raspberry Pi | 100.72.107.106  | Client only        |

## Environment Variables

### Cloud (PC)

```bash
# Model paths (optional - auto-detected)
export ROVY_TEXT_MODEL=/path/to/gemma.gguf
export ROVY_VISION_MODEL=/path/to/llava.gguf
export ROVY_VISION_MMPROJ=/path/to/mmproj.gguf

# Settings
export ROVY_GPU_LAYERS=-1        # -1 = all on GPU
export ROVY_WHISPER_MODEL=base   # tiny/base/small/medium
export ROVY_TTS_ENGINE=piper     # piper/espeak
```

### Robot (Raspberry Pi)

```bash
export ROVY_PC_IP=100.121.110.125    # Your PC's Tailscale IP
export ROVY_SERIAL_PORT=/dev/ttyACM0 # ESP32 connection
export ROVY_CAMERA_INDEX=0           # Camera device
```

## API Endpoints

### REST API (Mobile App)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Robot status |
| `/status` | GET | Battery, CPU, temp |
| `/camera/stream` | GET | MJPEG video stream |
| `/camera/snapshot` | GET | Single frame |
| `/control/move` | POST | Move robot |
| `/control/head` | POST | Gimbal control |
| `/face-recognition/recognize` | POST | Identify faces |
| `/wifi/scan` | GET | Scan networks |
| `/wifi/connect` | POST | Connect to WiFi |

### WebSocket (Robot)

Messages sent TO robot:
- `speak` - Play TTS audio
- `move` - Movement command
- `gimbal` - Camera head control
- `lights` - LED control

Messages FROM robot:
- `audio_data` - Microphone audio
- `image_data` - Camera frame
- `sensor_data` - Battery, IMU

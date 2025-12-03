"""
Rovy Cloud Server Configuration
Unified config for REST API (mobile app) + WebSocket (robot) + AI processing
"""
import os

# =============================================================================
# Network Configuration
# =============================================================================

# Server binding
HOST = "0.0.0.0"
API_PORT = 8000        # FastAPI REST for mobile app
WS_PORT = 8765         # WebSocket for robot

# Tailscale IPs (update these for your network)
PC_SERVER_IP = os.getenv("ROVY_PC_IP", "100.121.110.125")
ROBOT_IP = os.getenv("ROVY_ROBOT_IP", "100.72.107.106")

# =============================================================================
# AI Model Configuration (Local Models via llama.cpp)
# =============================================================================

# Text model (Gemma, Llama, Mistral)
TEXT_MODEL_PATH = os.getenv("ROVY_TEXT_MODEL", None)  # Auto-detect if None

# Vision model (LLaVA, Phi-3-Vision)
VISION_MODEL_PATH = os.getenv("ROVY_VISION_MODEL", None)
VISION_MMPROJ_PATH = os.getenv("ROVY_VISION_MMPROJ", None)

# Model settings
N_GPU_LAYERS = int(os.getenv("ROVY_GPU_LAYERS", "-1"))  # -1 = all on GPU
N_CTX = int(os.getenv("ROVY_CTX", "2048"))

# =============================================================================
# Speech Configuration
# =============================================================================

# Speech-to-text (Whisper)
WHISPER_MODEL = os.getenv("ROVY_WHISPER_MODEL", "base")  # tiny, base, small, medium, large

# Text-to-speech
TTS_ENGINE = os.getenv("ROVY_TTS_ENGINE", "piper")  # piper, espeak
PIPER_VOICE_PATH = os.getenv("ROVY_PIPER_VOICE", None)

# =============================================================================
# Assistant Configuration
# =============================================================================

ASSISTANT_NAME = "Rovy"
WAKE_WORDS = ["hey rovy", "rovy", "hey robot"]

# =============================================================================
# Face Recognition
# =============================================================================

KNOWN_FACES_DIR = os.getenv("ROVY_KNOWN_FACES", "known-faces")
FACE_RECOGNITION_THRESHOLD = float(os.getenv("ROVY_FACE_THRESHOLD", "0.6"))

# =============================================================================
# Robot Hardware (sent to robot client)
# =============================================================================

ROVER_SERIAL_PORT = "/dev/ttyACM0"
ROVER_BAUDRATE = 115200

# Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
JPEG_QUALITY = 80

# Audio
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024

# =============================================================================
# Connection Settings
# =============================================================================

RECONNECT_DELAY = 5
MAX_RECONNECT_ATTEMPTS = 0  # 0 = infinite

# =============================================================================
# Spotify Configuration
# =============================================================================

SPOTIFY_ENABLED = os.getenv("SPOTIFY_ENABLED", "true").lower() == "true"
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "93138e86ecf24daea4b07df74c7cb8e9")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "f8f131ad542a4cf2a021aae8bdbc5763")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
SPOTIFY_DEVICE_NAME = os.getenv("SPOTIFY_DEVICE_NAME", "ROVY")  # Raspotify device name

# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


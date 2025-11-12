"""
Smart Assistant for Rover
Integrates ReSpeaker mic array, speech recognition, and LLaVa vision-language model
"""
import os
import sys
import queue
import threading
import time
import numpy as np
import cv2
from datetime import datetime
import subprocess
import json
import tempfile
import wave
from contextlib import contextmanager

# Suppress ALSA warnings
os.environ['ALSA_CARD'] = 'ArrayUAC10'

@contextmanager
def suppress_alsa_warnings():
    """Temporarily suppress stderr to hide ALSA warnings."""
    stderr_fd = sys.stderr.fileno()
    with os.fdopen(os.dup(stderr_fd), 'w') as old_stderr:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                yield
            finally:
                os.dup2(old_stderr.fileno(), stderr_fd)

# Speech recognition
try:
    import speech_recognition as sr
except ImportError:
    print("⚠️  speech_recognition not installed. Install with: pip install SpeechRecognition")
    sr = None

# Audio playback
try:
    import pyaudio
except ImportError:
    print("⚠️  pyaudio not installed. Install with: pip install pyaudio")
    pyaudio = None

# ReSpeaker LED control
try:
    from pixel_ring import pixel_ring
    PIXEL_RING_AVAILABLE = True
except ImportError:
    print("⚠️  pixel_ring not available. Visual feedback will be limited.")
    PIXEL_RING_AVAILABLE = False


class ReSpeakerMicrophone(sr.Microphone):
    """Custom Microphone class for ReSpeaker - reads 6 channels, extracts channel 0 for mono."""
    
    def __init__(self, device_index=None, sample_rate=16000, chunk_size=1024):
        """Initialize with ReSpeaker-specific settings."""
        self.device_index = device_index
        self.format = 8  # paInt16
        self.SAMPLE_WIDTH = 2
        self.SAMPLE_RATE = sample_rate
        self.CHUNK = chunk_size
        self.audio = None
        self.stream = None
        # ReSpeaker has 6 channels - read all, extract channel 0
        self.CHANNELS = 6
    
    def __enter__(self):
        """Open the audio stream with 6 channels."""
        assert self.stream is None, "This audio source is already inside a context manager"
        
        with suppress_alsa_warnings():
            self.audio = pyaudio.PyAudio()
        
        try:
            with suppress_alsa_warnings():
                self.stream = self.audio.open(
                    input_device_index=self.device_index,
                    channels=self.CHANNELS,  # Read all 6 channels
                    format=self.format,
                    rate=self.SAMPLE_RATE,
                    frames_per_buffer=self.CHUNK,
                    input=True,
                )
            # Wrap stream to extract only channel 0
            self.stream.read = self._make_channel_0_reader(self.stream.read)
        except Exception as e:
            self.audio.terminate()
            raise e
        return self
    
    def _make_channel_0_reader(self, original_read):
        """Wrap the read function to extract only channel 0 from 6-channel audio."""
        def read_mono(size):
            # Read 6-channel data
            data = original_read(size)
            # Convert to numpy array (int16)
            audio_data = np.frombuffer(data, dtype=np.int16)
            # Reshape to (samples, 6 channels)
            audio_data = audio_data.reshape(-1, 6)
            # Extract only channel 0
            mono_data = audio_data[:, 0]
            # Convert back to bytes
            return mono_data.tobytes()
        return read_mono
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Close the audio stream."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None


class ReSpeakerInterface:
    """Interface for ReSpeaker mic array with voice activity detection."""
    
    def __init__(self, device_index=None):
        """
        Initialize ReSpeaker microphone.
        
        Args:
            device_index: Audio device index (None = auto-detect)
        """
        if sr is None:
            raise RuntimeError("speech_recognition not installed")
        
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.device_index = device_index
        
        # Try to find ReSpeaker device
        if device_index is None:
            self.device_index = self._find_respeaker()
        
        # Configure recognizer for better performance
        self.recognizer.energy_threshold = 300  # Adjust based on environment
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Seconds of silence to consider end of phrase
        
        print(f"[ReSpeaker] Initialized (device index: {self.device_index})")
        
        # Initialize LED ring if available
        self.led_available = False
        if PIXEL_RING_AVAILABLE:
            try:
                pixel_ring.set_brightness(10)
                pixel_ring.off()
                self.led_available = True
            except Exception as e:
                print(f"[ReSpeaker] LED ring init failed: {e}")
                print(f"[ReSpeaker] Continuing without LED feedback (run with sudo for LED access)")
    
    def _find_respeaker(self):
        """Auto-detect ReSpeaker device index."""
        if pyaudio is None:
            return None
        
        with suppress_alsa_warnings():
            p = pyaudio.PyAudio()
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                name = info.get('name', '').lower()
                if 'respeaker' in name or 'seeed' in name:
                    print(f"[ReSpeaker] Found device: {info.get('name')} at index {i}")
                    p.terminate()
                    return i
            p.terminate()
        
        print("[ReSpeaker] ReSpeaker device not found, using default microphone")
        return None
    
    def listen(self, timeout=None, phrase_time_limit=10):
        """
        Listen for speech and return transcribed text.
        
        Args:
            timeout: Maximum time to wait for speech to start (None = infinite)
            phrase_time_limit: Maximum duration of a phrase
            
        Returns:
            str: Transcribed text, or None if no speech detected
        """
        if self.microphone is None:
            # Use custom ReSpeaker microphone class that forces 1 channel (mono)
            self.microphone = ReSpeakerMicrophone(
                device_index=self.device_index,
                sample_rate=16000,
                chunk_size=1024
            )
        
        try:
            # LED: Listening - Blue color
            self.set_led('listen')
            
            print("[ReSpeaker] 🎤 Listening...")
            
            try:
                with self.microphone as source:
                    # Adjust for ambient noise quickly
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    
                    # Listen for audio
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_time_limit
                    )
            except OSError as e:
                # Microphone channel/configuration error
                print(f"[ReSpeaker] ❌ Microphone configuration error: {e}")
                print("[ReSpeaker] 💡 Tip: ReSpeaker may need specific ALSA configuration")
                self.set_led('off')
                return None
            
            # LED: Processing - Orange/Yellow color
            self.set_led('think')
            
            print("[ReSpeaker] 🔄 Processing audio...")
            
            # Recognize speech using Google Speech Recognition
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"[ReSpeaker] 📝 Heard: '{text}'")
                
                # LED: Turn off after recognition
                self.set_led('off')
                
                return text
            
            except sr.UnknownValueError:
                print("[ReSpeaker] ❌ Could not understand audio")
                self.set_led('off')
                return None
            
            except sr.RequestError as e:
                print(f"[ReSpeaker] ❌ Speech recognition service error: {e}")
                # Try offline recognition as fallback
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    print(f"[ReSpeaker] 📝 Heard (offline): '{text}'")
                    self.set_led('off')
                    return text
                except:
                    self.set_led('off')
                    return None
        
        except sr.WaitTimeoutError:
            print("[ReSpeaker] ⏰ Listening timeout")
            self.set_led('off')
            return None
        
        except Exception as e:
            print(f"[ReSpeaker] ❌ Error: {e}")
            self.set_led('off')
            return None
    
    def set_led(self, mode):
        """
        Set LED ring mode with distinct colors for different states.
        
        Args:
            mode: 'off', 'listen', 'think', 'speak', or tuple (r, g, b) for custom color
        """
        if not PIXEL_RING_AVAILABLE or not self.led_available:
            return
        
        try:
            if mode == 'off':
                pixel_ring.off()
            elif mode == 'listen':
                # Blue/Cyan for listening - indicates it's ready to hear
                pixel_ring.set_color(r=0, g=150, b=255)  # Bright blue
            elif mode == 'think':
                # Yellow/Orange for thinking - indicates processing
                pixel_ring.set_color(r=255, g=150, b=0)  # Orange/yellow
            elif mode == 'speak':
                # Green for speaking - indicates responding
                pixel_ring.set_color(r=0, g=255, b=0)  # Green
            elif isinstance(mode, tuple) and len(mode) == 3:
                # Custom RGB color
                r, g, b = mode
                pixel_ring.set_color(r=r, g=g, b=b)
            else:
                # Default to off for unknown modes
                pixel_ring.off()
        except Exception as e:
            # Silently fail if LED control doesn't work
            pass


class LLaVaAssistant:
    """Interface to LLaVa vision-language model via llama.cpp."""
    
    def __init__(self, model_path=None, mmproj_path=None):
        """
        Initialize LLaVa assistant.
        
        Args:
            model_path: Path to LLaVa model (GGUF format)
            mmproj_path: Path to multimodal projector
        """
        # Auto-detect common paths if not specified
        if model_path is None:
            possible_paths = [
                "/home/jetson/models/llava-v1.5-7b-Q4_K_M.gguf",
                "/home/jetson/llava/models/llava-v1.5-7b-Q4_K_M.gguf",
                "./models/llava-v1.5-7b-Q4_K_M.gguf",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if mmproj_path is None:
            possible_paths = [
                "/home/jetson/models/mmproj-model-f16.gguf",
                "/home/jetson/llava/models/mmproj-model-f16.gguf",
                "./models/mmproj-model-f16.gguf",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    mmproj_path = path
                    break
        
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        
        # Find llama.cpp executable
        self.llava_cli = self._find_llava_cli()
        
        if not self.llava_cli:
            print("⚠️  Warning: llama.cpp llava-cli not found!")
            print("    Looking for: llava-cli, llama-llava-cli, or ./llama.cpp/build/bin/llava-cli")
        
        print(f"[LLaVa] Model: {model_path}")
        print(f"[LLaVa] MMProj: {mmproj_path}")
        print(f"[LLaVa] CLI: {self.llava_cli}")
    
    def _find_llava_cli(self):
        """Find llama.cpp llava executable."""
        possible_names = [
            "/home/jetson/llama.cpp/build/bin/llama-cli",  # Text-only mode
            "/home/jetson/llama.cpp/build/bin/llama-mtmd-cli",  # Multimodal
            "llama-cli",
            "llama-mtmd-cli",
            "/usr/local/bin/llama-cli",
        ]
        
        for name in possible_names:
            try:
                # Check if command exists
                if name.startswith("/"):
                    # Absolute path - check if file exists
                    if os.path.exists(name):
                        return name
                else:
                    # Command name - use which
                    result = subprocess.run(
                        ["which", name],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        return name
            except:
                pass
        
        return None
    
    def ask(self, question, image=None, max_tokens=512, temperature=0.7, realtime_context=None):
        """
        Ask LLaVa a question, optionally with an image.
        
        Args:
            question: Question text
            image: OpenCV image (numpy array), or None for text-only
            max_tokens: Maximum response length
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            realtime_context: Optional string with real-time info (time, weather, etc.)
            
        Returns:
            str: Model's response
        """
        if not self.llava_cli or not self.model_path:
            return "Error: LLaVa not properly configured"
        
        print(f"[LLaVa] Question: {question}")
        
        # Determine if we're using text-only or multimodal mode
        use_vision = image is not None and self.mmproj_path is not None
        is_mtmd = "mtmd" in self.llava_cli
        
        # Save image to temporary file if provided
        image_path = None
        if use_vision:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image_path = tmp.name
                cv2.imwrite(image_path, image)
        
        try:
            # Build prompt with real-time context - let LLM understand naturally
            # Add real-time context if provided (includes time, weather, etc.)
            realtime_section = ""
            if realtime_context:
                realtime_section = f"\n\nReal-time information available:\n{realtime_context}"
            
            # Build prompt with context - make responses natural and direct
            system_context = f"""You are Rovy, a helpful and friendly robot assistant.{realtime_section}

IMPORTANT IDENTITY RULES:
- You are Rovy, a robot assistant - NOT a language model or AI
- NEVER say "As a language model", "As an AI", or similar phrases
- When asked who you are or what you are, say "I'm Rovy, a robot assistant"
- Be confident in your identity as Rovy

Instructions:
- Give direct, natural answers like a human would
- When asked about time, date, weather, or current information, use the real-time data provided above
- Be conversational and helpful
- Keep answers concise but complete
- Always identify yourself as Rovy when relevant"""
            
            if use_vision and is_mtmd:
                prompt = f"{system_context}\n\nUSER: {question}\nASSISTANT:"
            else:
                prompt = f"{system_context}\n\nUSER: {question}\nASSISTANT:"
            
            # Build command - EXTREME SPEED optimizations
            cmd = [
                self.llava_cli,
                "-m", self.model_path,
                "-p", prompt,
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "-ngl", "33",  # GPU layers (all 33 layers on GPU)
                "-c", "512",   # Minimal context (was 1024)
                "-b", "256",   # Smaller batch (faster)
                "--n-predict", str(max_tokens),  # Hard limit
            ]
            
            # Add vision components if using multimodal mode
            if use_vision and is_mtmd:
                cmd.extend([
                    "--mmproj", self.mmproj_path,
                    "--image", image_path,
                    "--chat-template", "vicuna",
                ])
            
            print(f"[LLaVa] Running inference (mode: {'vision' if use_vision else 'text-only'})...")
            
            # Run inference with timing
            import time
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                print(f"[LLaVa] Error: {result.stderr[:200]}")
                return "I encountered an error processing your request."
            
            print(f"[LLaVa] ⚡ Inference completed in {elapsed:.1f}s")
            
            # Parse output
            response = result.stdout.strip()
            
            # Clean up response
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            if "USER:" in response:
                response = response.split("USER:")[0].strip()
            
            # Remove prompt echo if present
            if question in response:
                response = response.replace(question, "").strip()
            
            # Remove common artifacts
            response = response.replace("[end of text]", "").strip()
            response = response.replace("[End of text]", "").strip()
            response = response.replace("</s>", "").strip()
            
            # Clean up extra whitespace
            response = " ".join(response.split())
            
            print(f"[LLaVa] Response: {response[:100]}...")
            return response if response else "I'm not sure how to answer that."
        
        except subprocess.TimeoutExpired:
            print("[LLaVa] ⏰ Inference timeout")
            return "Sorry, that took too long to process."
        
        except Exception as e:
            print(f"[LLaVa] ❌ Error: {e}")
            return f"Error: {str(e)}"
        
        finally:
            # Clean up temporary image
            if image_path and os.path.exists(image_path):
                try:
                    os.unlink(image_path)
                except:
                    pass


class TextToSpeech:
    """Simple text-to-speech using espeak or piper, with print-only fallback."""
    
    def __init__(self, engine='espeak', print_only=False):
        """
        Initialize TTS engine.
        
        Args:
            engine: 'espeak' (fast, robotic), 'piper' (natural, slower), or 'print' (text only)
            print_only: If True, only print text instead of speaking
        """
        self.print_only = print_only
        self.engine = engine
        self.piper_voice = None
        self.audio_device = None  # Will be auto-detected
        
        if print_only or engine == 'print':
            print("[TTS] Using print-only mode (no speakers)")
            self.engine = 'print'
            return
        
        # Auto-detect audio output device
        self._detect_audio_device()
        
        # Check if engine is available
        if engine == 'espeak':
            try:
                subprocess.run(['espeak', '--version'], capture_output=True, timeout=2)
                print("[TTS] Using espeak")
            except:
                print("⚠️  espeak not found or no speakers - using print-only mode")
                self.engine = 'print'
        elif engine == 'piper':
            try:
                from piper import PiperVoice
                # Load LIBRITTS HIGH QUALITY - premium audiobook quality, most natural Piper voice
                print("[TTS] Loading LIBRITTS HIGH QUALITY voice (premium natural voice)...")
                voice_path = os.path.expanduser("~/.local/share/piper-voices/en_US-libritts-high.onnx")
                # Fallback to lessac if libritts not available
                if not os.path.exists(voice_path):
                    voice_path = os.path.expanduser("~/.local/share/piper-voices/en_US-lessac-high.onnx")
                    if not os.path.exists(voice_path):
                        print(f"⚠️  High quality voice not found")
                        print("⚠️  Falling back to espeak")
                        self.engine = 'espeak'
                        return
                    print("[TTS] Using Lessac High (LibriTTS not found)")
                else:
                    print("[TTS] Using LibriTTS High (premium quality)")
                
                self.piper_voice = PiperVoice.load(
                    voice_path,
                    use_cuda=False  # Use CPU on Jetson
                )
                print("[TTS] ✅ Natural human voice ready!")
            except Exception as e:
                print(f"⚠️  Piper not available ({e}). Falling back to espeak")
                self.engine = 'espeak'
    
    def _detect_audio_device(self):
        """Auto-detect USB audio output device."""
        try:
            # List all audio devices
            result = subprocess.run(
                ['aplay', '-l'],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode != 0:
                print("[TTS] ⚠️  Could not list audio devices")
                self.audio_device = None
                return
            
            # Look for USB audio devices (excluding ReSpeaker which is input)
            lines = result.stdout.split('\n')
            for line in lines:
                # Look for USB audio devices that are not ReSpeaker
                if 'card' in line and 'USB Audio' in line:
                    # Extract card number - exclude ReSpeaker/ArrayUAC (those are input mics)
                    if 'ReSpeaker' not in line and 'ArrayUAC' not in line:
                        # Found a USB audio device (likely speakers)
                        parts = line.split()
                        card_num = None
                        device_num = '0'  # Default device number
                        
                        # Extract card number
                        for i, part in enumerate(parts):
                            if part == 'card' and i + 1 < len(parts):
                                card_num = parts[i + 1].rstrip(':')
                                break
                        
                        # Extract device number if specified
                        for i, part in enumerate(parts):
                            if part == 'device' and i + 1 < len(parts):
                                device_num = parts[i + 1].rstrip(':')
                                break
                        
                        if card_num:
                            # Use plughw for format conversion
                            device_str = f'plughw:{card_num},{device_num}'
                            # Just use it - if it doesn't work, we'll fall back to default when playing
                            self.audio_device = device_str
                            print(f"[TTS] ✅ Detected audio output: {device_str} ({line.strip()})")
                            return
            
            # If no USB device found, try default
            print("[TTS] ⚠️  No USB audio device found, will use default")
            self.audio_device = None
            
        except Exception as e:
            print(f"[TTS] ⚠️  Error detecting audio device: {e}")
            import traceback
            print(f"[TTS] Traceback: {traceback.format_exc()}")
            self.audio_device = None
    
    def _preprocess_text(self, text):
        """
        Preprocess text for more natural TTS pronunciation.
        Converts numbers, expands abbreviations, adds natural pauses.
        """
        import re
        
        # Convert numbers to words for better pronunciation
        def number_to_words(num_str):
            """Convert number string to words"""
            try:
                num = int(num_str)
                if num < 20:
                    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
                            'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
                            'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
                    return words[num] if num < len(words) else num_str
                elif num < 100:
                    tens = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
                    ones = num % 10
                    ten = num // 10
                    if ones == 0:
                        return tens[ten - 2]
                    return f"{tens[ten - 2]} {number_to_words(str(ones))}"
                else:
                    return num_str  # Keep large numbers as digits
            except:
                return num_str
        
        # Replace standalone numbers (1-99) with words
        text = re.sub(r'\b(\d{1,2})\b', lambda m: number_to_words(m.group(1)), text)
        
        # Expand common abbreviations
        abbreviations = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bMs\.': 'Miss',
            r'\bProf\.': 'Professor',
            r'\bvs\.': 'versus',
            r'\betc\.': 'etcetera',
            r'\bi\.e\.': 'that is',
            r'\be\.g\.': 'for example',
        }
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Add natural pauses after punctuation
        text = re.sub(r'([.!?])\s+', r'\1 ', text)  # Ensure space after sentence enders
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _wait_for_audio_device(self, device=None, max_attempts=10, delay=0.5):
        """Wait for audio device to be available."""
        # Use detected device or default
        if device is None:
            device = self.audio_device if self.audio_device else 'default'
        
        for attempt in range(max_attempts):
            try:
                # Test if device is available by listing devices
                result = subprocess.run(
                    ['aplay', '-l'],
                    capture_output=True,
                    timeout=2
                )
                if result.returncode == 0:
                    # If using default, just check that aplay works
                    if device == 'default':
                        return True
                    
                    # Test the specific device
                    test_result = subprocess.run(
                        ['aplay', '-D', device, '--dump-hw-params'],
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        timeout=2
                    )
                    if test_result.returncode == 0:
                        return True
            except Exception as e:
                pass
            time.sleep(delay)
        return False
    
    def speak(self, text, speed=150):
        """
        Speak text aloud (or print if no speakers).
        
        Args:
            text: Text to speak
            speed: Speech speed (words per minute) - used for espeak only
        """
        if not text:
            return
        
        if self.engine == 'print' or self.print_only:
            print(f"\n[ASSISTANT] 🔊 {text}\n")
            return
        
        # Preprocess text for natural pronunciation
        processed_text = self._preprocess_text(text)
        
        print(f"[TTS] 🔊 Speaking: '{processed_text[:50]}...'")
        
        # Wait for audio device to be ready (especially important on startup)
        if not self._wait_for_audio_device():
            print("[TTS] ⚠️  Audio device not ready, trying anyway...")
        
        try:
            if self.engine == 'espeak':
                # Write to temp file, then play to USB speakers (plughw:2,0)
                # This ensures proper routing to USB speakers
                subprocess.run(
                    ['espeak', '-w', '/tmp/speech.wav', '-s', str(speed), processed_text],
                    capture_output=True,
                    timeout=30
                )
                # Play to detected audio device or default
                device = self.audio_device if self.audio_device else 'default'
                if device != 'default':
                    aplay_cmd = ['aplay', '-D', device, '/tmp/speech.wav']
                else:
                    aplay_cmd = ['aplay', '/tmp/speech.wav']
                
                aplay_result = subprocess.run(
                    aplay_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if aplay_result.returncode != 0:
                    print(f"[TTS] ⚠️  aplay error ({device}): {aplay_result.stderr}")
                    # Try default device as fallback
                    if device != 'default':
                        print("[TTS] Trying default audio device...")
                        default_result = subprocess.run(
                            ['aplay', '/tmp/speech.wav'],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if default_result.returncode != 0:
                            print(f"[TTS] ⚠️  Default device also failed: {default_result.stderr}")
                else:
                    print(f"[TTS] ✅ Audio played successfully on {device}")
                # Clean up temp file
                try:
                    os.unlink('/tmp/speech.wav')
                except:
                    pass
            elif self.engine == 'piper':
                # Piper produces high-quality, natural-sounding speech
                if self.piper_voice:
                    from piper.config import SynthesisConfig
                    
                    # Configure for MAXIMUM naturalness - optimized for LibriTTS
                    config = SynthesisConfig(
                        noise_scale=0.667,      # Default for LibriTTS (more stable than 0.8)
                        length_scale=1.0,       # Normal speed (1.0 = natural pace)
                        noise_w_scale=0.8,      # Phoneme duration variation (adds natural rhythm)
                        volume=1.2              # Slightly louder for clarity
                    )
                    
                    # Use preprocessed text for better pronunciation
                    # Synthesize speech with natural parameters
                    audio_chunks = []
                    for chunk in self.piper_voice.synthesize(processed_text, syn_config=config):
                        audio_chunks.append(chunk.audio_int16_bytes)
                    
                    audio_data = b''.join(audio_chunks)
                    
                    # Write WAV file
                    with wave.open('/tmp/speech.wav', 'wb') as wav_file:
                        wav_file.setnchannels(1)    # Mono
                        wav_file.setsampwidth(2)    # 16-bit
                        wav_file.setframerate(22050)  # 22.05 kHz
                        wav_file.writeframes(audio_data)
                    
                    # Play with aplay to detected audio device or default
                    device = self.audio_device if self.audio_device else 'default'
                    if device != 'default':
                        aplay_cmd = ['aplay', '-D', device, '/tmp/speech.wav']
                    else:
                        aplay_cmd = ['aplay', '/tmp/speech.wav']
                    
                    aplay_result = subprocess.run(
                        aplay_cmd,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if aplay_result.returncode != 0:
                        print(f"[TTS] ⚠️  aplay error ({device}): {aplay_result.stderr}")
                        # Try default device as fallback
                        if device != 'default':
                            print("[TTS] Trying default audio device...")
                            default_result = subprocess.run(
                                ['aplay', '/tmp/speech.wav'],
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            if default_result.returncode != 0:
                                print(f"[TTS] ⚠️  Default device also failed: {default_result.stderr}")
                    else:
                        print(f"[TTS] ✅ Audio played successfully on {device}")
                    # Clean up temp file
                    try:
                        os.unlink('/tmp/speech.wav')
                    except:
                        pass
        
        except Exception as e:
            print(f"[TTS] ❌ Error: {e}")
            print(f"[ASSISTANT] 🔊 {text}")


class RealTimeInfo:
    """Helper class to get real-time information like time, weather, etc."""
    
    def __init__(self, weather_api_key=None):
        """
        Initialize real-time info helper.
        
        Args:
            weather_api_key: OpenWeatherMap API key (optional, for weather)
        """
        self.weather_api_key = weather_api_key
        self.weather_cache = {}
        self.weather_cache_time = 0
        self.weather_cache_duration = 300  # Cache weather for 5 minutes
    
    def get_current_time(self, format_type='full'):
        """Get current time in various formats."""
        now = datetime.now()
        
        if format_type == 'time':
            return now.strftime("%I:%M %p")
        elif format_type == 'date':
            return now.strftime("%A, %B %d, %Y")
        elif format_type == 'full':
            return now.strftime("%A, %B %d, %Y at %I:%M %p")
        elif format_type == 'simple':
            return now.strftime("%I:%M %p on %B %d")
        else:
            return now.strftime("%A, %B %d, %Y at %I:%M %p")
    
    def get_weather(self, city="auto", units="metric"):
        """
        Get current weather information.
        
        Args:
            city: City name or "auto" to try to detect
            units: "metric" (Celsius) or "imperial" (Fahrenheit)
        """
        if not self.weather_api_key:
            return None
        
        # Check cache
        cache_key = f"{city}_{units}"
        if time.time() - self.weather_cache_time < self.weather_cache_duration:
            if cache_key in self.weather_cache:
                return self.weather_cache[cache_key]
        
        try:
            import urllib.request
            import json
            
            # Try to auto-detect city from IP if "auto"
            if city == "auto":
                try:
                    # Get approximate location from IP
                    with urllib.request.urlopen('http://ip-api.com/json/', timeout=3) as response:
                        location_data = json.loads(response.read().decode())
                        city = location_data.get('city', 'London')  # Default fallback
                except:
                    city = "London"  # Default fallback
            
            # Get weather from OpenWeatherMap
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units={units}"
            
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode())
                
                temp = data['main']['temp']
                description = data['weather'][0]['description']
                humidity = data['main']['humidity']
                temp_unit = "°C" if units == "metric" else "°F"
                
                weather_info = {
                    'temperature': temp,
                    'description': description,
                    'humidity': humidity,
                    'unit': temp_unit,
                    'city': data['name']
                }
                
                # Cache it
                self.weather_cache[cache_key] = weather_info
                self.weather_cache_time = time.time()
                
                return weather_info
        except Exception as e:
            print(f"[Weather] ⚠️  Error fetching weather: {e}")
            return None
    
    def get_context_string(self):
        """Get formatted context string with all available real-time info."""
        context_parts = []
        
        # Always include time
        current_time = self.get_current_time('full')
        context_parts.append(f"Current date and time: {current_time}")
        
        # Include weather if available
        if self.weather_api_key:
            weather = self.get_weather()
            if weather:
                temp = int(weather['temperature'])
                desc = weather['description']
                city = weather['city']
                unit = weather['unit']
                context_parts.append(f"Current weather in {city}: {desc}, {temp}{unit}")
        
        return ". ".join(context_parts) + "."


class SmartAssistant:
    """
    Smart assistant combining speech recognition, vision, and LLaVa.
    """
    
    def __init__(self, camera=None, motor_controller=None, llava_model_path=None, llava_mmproj_path=None, print_only=False, weather_api_key=None):
        """
        Initialize smart assistant.
        
        Args:
            camera: Camera object with capture_frames() method (e.g., OakDDepthCamera)
            motor_controller: Motor controller object for movement commands
            llava_model_path: Path to LLaVa model
            llava_mmproj_path: Path to LLaVa multimodal projector
            print_only: If True, print responses instead of speaking (no speakers)
        """
        self.camera = camera
        self.motor_controller = motor_controller
        self.print_only = print_only
        
        # Initialize components
        print("[Assistant] Initializing smart assistant...")
        
        # Auto-detect model paths if not provided
        if llava_model_path is None:
            possible_paths = [
                "/home/jetson/.cache/llava-v1.5-7b-q4.gguf",
                "/home/jetson/.cache/llava-v1.5-7b-q5.gguf",
                "/home/jetson/models/llava-v1.5-7b-Q4_K_M.gguf",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    llava_model_path = path
                    break
        
        if llava_mmproj_path is None:
            possible_paths = [
                "/home/jetson/.cache/llava-mmproj-fixed.gguf",
                "/home/jetson/.cache/llava-mmproj-f16.gguf",
                "/home/jetson/models/mmproj-model-f16.gguf",
            ]
            for path in possible_paths:
                if os.path.exists(path) and os.path.getsize(path) > 1000:  # Not empty
                    llava_mmproj_path = path
                    break
        
        self.respeaker = ReSpeakerInterface()
        self.llava = LLaVaAssistant(llava_model_path, llava_mmproj_path)
        # Use piper for natural human voice, falls back to espeak if piper unavailable
        self.tts = TextToSpeech(engine='print' if print_only else 'piper', print_only=print_only)
        # Real-time information helper
        self.realtime_info = RealTimeInfo(weather_api_key=weather_api_key)
        
        # Wake word detection
        self.wake_words = ['hey rover', 'hey robot', 'rover', 'hey rovy', 'rovy']
        
        # Conversation state
        self.is_active = False
        self.last_interaction = time.time()
        
        # Automatic navigation process tracking
        self.auto_nav_process = None
        self.auto_nav_script_path = os.path.join(os.path.dirname(__file__), 'depth_llava_nav.py')
        
        print("[Assistant] ✅ Smart assistant ready!")
        if print_only:
            print("[Assistant] 💡 Running in print-only mode (no speakers)")
    
    def _check_wake_word(self, text):
        """Check if text contains wake word."""
        if not text:
            return False
        
        text_lower = text.lower()
        for wake_word in self.wake_words:
            if wake_word in text_lower:
                return True
        return False
    
    def _remove_wake_word(self, text):
        """Remove wake word from text."""
        if not text:
            return text
        
        text_lower = text.lower()
        for wake_word in self.wake_words:
            if wake_word in text_lower:
                # Remove wake word and clean up
                text = text_lower.replace(wake_word, '').strip()
                # Remove leading question words if isolated
                text = text.lstrip('?,. ')
                return text
        return text
    
    def listen_for_wake_word(self, timeout=5):
        """
        Listen for wake word.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            bool: True if wake word detected
        """
        text = self.respeaker.listen(timeout=timeout)
        
        if self._check_wake_word(text):
            print("[Assistant] 👂 Wake word detected!")
            self.respeaker.set_led('speak')
            self.tts.speak("Yes?")
            self.respeaker.set_led('off')
            return True
        
        return False
    
    def _check_movement_command(self, text):
        """
        Check if text contains a movement command.
        
        Returns:
            dict: {'action': str, 'duration': float} or None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Check for automatic navigation commands first
        auto_nav_keywords = [
            'automatic movement', 'auto movement', 'autonomous movement',
            'show automatic movement', 'start automatic', 'start autonomous',
            'automatic navigation', 'auto navigation', 'autonomous navigation',
            'start navigation', 'begin navigation'
        ]
        for keyword in auto_nav_keywords:
            if keyword in text_lower:
                return {'action': 'auto_nav', 'duration': 0}
        
        # Check for stop command (can stop both manual and automatic movement)
        if any(word in text_lower for word in ['stop', 'halt', 'freeze']):
            return {'action': 'stop', 'duration': 0}
        
        # Movement command patterns
        commands = {
            'forward': ['forward', 'go forward', 'move forward', 'go ahead', 'straight'],
            'backward': ['backward', 'back', 'go back', 'move back', 'reverse'],
            'left': ['left', 'turn left', 'go left'],
            'right': ['right', 'turn right', 'go right'],
            # Battery/status check
            'status': ['battery', 'power', 'charge', 'status'],
        }
        
        # Check for commands
        for action, keywords in commands.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Try to extract duration (e.g., "forward 2 seconds")
                    duration = 3.0  # Default 3 seconds = ~1 meter movement
                    import re
                    match = re.search(r'(\d+)\s*(second|meter|m)', text_lower)
                    if match:
                        duration = float(match.group(1))
                    
                    return {'action': action, 'duration': duration}
        
        return None
    
    def _execute_movement(self, command):
        """Execute a movement command on the motor controller."""
        action = command['action']
        duration = command['duration']
        
        try:
            # Handle automatic navigation start
            if action == 'auto_nav':
                return self._start_automatic_navigation()
            
            # Handle stop command (can stop both manual movement and automatic navigation)
            if action == 'stop':
                return self._stop_movement()
            
            # Manual movement commands require motor controller
            if not self.motor_controller:
                return "I don't have motor control available."
            
            # Handle status check
            if action == 'status':
                return "Battery monitoring not available. If rover moves weakly, charge the battery or use power adapter."
            
            print(f"[Motors] 🚗 Executing: {action} for {duration}s")
            
            # Use same low-level motor control as depth_llava_nav.py
            # Use stronger speeds for noticeable movement (rover_controller uses 0.4-0.7)
            speed_val = 0.5  # Strong medium speed
            
            # Calculate motor power for each direction (like depth_llava_nav.py)
            if action == 'forward':
                L = speed_val
                R = speed_val
            elif action == 'backward':
                L = -speed_val
                R = -speed_val
            elif action == 'left':
                turn_power = 0.5  # Strong turn
                L = -turn_power
                R = turn_power
            elif action == 'right':
                turn_power = 0.5  # Strong turn
                L = turn_power
                R = -turn_power
            else:
                L = 0.0
                R = 0.0
            
            # Send motor commands for the specified duration
            if hasattr(self.motor_controller, '_send'):
                print(f"[Motors] 🔧 Sending commands: L={L:.2f}, R={R:.2f} for {duration}s")
                start_time = time.time()
                command_count = 0
                
                while (time.time() - start_time) < duration:
                    self.motor_controller._send(L, R)
                    command_count += 1
                    time.sleep(0.1)  # 10Hz control rate
                
                elapsed = time.time() - start_time
                print(f"[Motors] ✅ Sent {command_count} commands over {elapsed:.2f}s")
                
                # Stop after duration
                self.motor_controller.stop()
                return f"Moved {action} for {duration:.1f} seconds"
            else:
                return "Motor control method (_send) not found"
        
        except Exception as e:
            print(f"[Motors] ❌ Error: {e}")
            # Make sure to stop on error
            try:
                self.motor_controller.stop()
            except:
                pass
            return f"Motor error: {str(e)}"
    
    def _start_automatic_navigation(self):
        """Start the automatic navigation script (depth_llava_nav.py)."""
        # Check if already running
        if self.auto_nav_process is not None:
            if self.auto_nav_process.poll() is None:
                return "Automatic navigation is already running."
            else:
                # Process finished, clean up
                self.auto_nav_process = None
        
        # Check if script exists
        if not os.path.exists(self.auto_nav_script_path):
            return f"Automatic navigation script not found at {self.auto_nav_script_path}"
        
        try:
            print(f"[AutoNav] 🚀 Starting automatic navigation...")
            # Start the script as a subprocess
            # Use the same port as the motor controller if available
            port = '/dev/ttyACM0'
            if self.motor_controller and hasattr(self.motor_controller, 'port'):
                port = self.motor_controller.port
            
            # Start the process in the background
            # Use subprocess.DEVNULL for output to prevent blocking, or log to files
            log_dir = os.path.join(os.path.dirname(self.auto_nav_script_path), 'logs')
            os.makedirs(log_dir, exist_ok=True)
            stdout_file = os.path.join(log_dir, 'autonav_stdout.log')
            stderr_file = os.path.join(log_dir, 'autonav_stderr.log')
            
            with open(stdout_file, 'w') as fout, open(stderr_file, 'w') as ferr:
                self.auto_nav_process = subprocess.Popen(
                    [sys.executable, self.auto_nav_script_path, '--port', port],
                    stdout=fout,
                    stderr=ferr,
                    cwd=os.path.dirname(self.auto_nav_script_path)
                )
            
            # Give it a moment to start, then check if it's still running
            time.sleep(0.5)
            if self.auto_nav_process.poll() is not None:
                # Process already exited - read error output
                error_msg = "Process exited immediately"
                try:
                    if os.path.exists(stderr_file):
                        with open(stderr_file, 'r') as f:
                            error_content = f.read()
                            if error_content:
                                error_msg = f"Process exited: {error_content[:200]}"
                except:
                    pass
                self.auto_nav_process = None
                print(f"[AutoNav] ❌ Failed to start: {error_msg}")
                return f"Failed to start automatic navigation: {error_msg}"
            
            print(f"[AutoNav] ✅ Started (PID: {self.auto_nav_process.pid})")
            print(f"[AutoNav] Logs: {stdout_file} and {stderr_file}")
            return "Starting automatic navigation. Say 'stop' to stop it."
        
        except Exception as e:
            print(f"[AutoNav] ❌ Error starting automatic navigation: {e}")
            self.auto_nav_process = None
            return f"Failed to start automatic navigation: {str(e)}"
    
    def _stop_movement(self):
        """Stop both manual movement and automatic navigation."""
        result_messages = []
        
        # Stop manual movement if motor controller is available
        if self.motor_controller:
            try:
                self.motor_controller.stop()
                result_messages.append("Stopped manual movement")
            except Exception as e:
                print(f"[Motors] Error stopping: {e}")
        
        # Stop automatic navigation if running
        if self.auto_nav_process is not None:
            if self.auto_nav_process.poll() is None:
                # Process is still running, terminate it
                try:
                    print(f"[AutoNav] 🛑 Stopping automatic navigation (PID: {self.auto_nav_process.pid})...")
                    self.auto_nav_process.terminate()
                    
                    # Wait a bit for graceful shutdown
                    try:
                        self.auto_nav_process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't stop
                        print("[AutoNav] Force killing process...")
                        self.auto_nav_process.kill()
                        self.auto_nav_process.wait()
                    
                    self.auto_nav_process = None
                    result_messages.append("Stopped automatic navigation")
                    print("[AutoNav] ✅ Stopped")
                except Exception as e:
                    print(f"[AutoNav] ❌ Error stopping: {e}")
                    result_messages.append(f"Error stopping automatic navigation: {str(e)}")
            else:
                # Process already finished
                self.auto_nav_process = None
        
        if result_messages:
            return ". ".join(result_messages) + "."
        else:
            return "Nothing to stop."
    
    def process_question(self, question, use_vision=True, max_tokens=150):
        """
        Process a question using LLaVa.
        
        Args:
            question: Question text
            use_vision: Whether to include camera image
            max_tokens: Maximum response length (default 150 for complete answers)
            
        Returns:
            str: Response
        """
        image = None
        
        # Capture image if vision is requested and camera available
        if use_vision and self.camera:
            try:
                rgb_frame, _ = self.camera.capture_frames()
                image = rgb_frame
                print("[Assistant] 📸 Using camera image for context")
            except Exception as e:
                print(f"[Assistant] ⚠️  Could not capture image: {e}")
        
        # Get real-time context (time, weather, etc.) - let LLM understand naturally
        realtime_context = self.realtime_info.get_context_string()
        
        # Get response from LLaVa with natural temperature and real-time context
        response = self.llava.ask(question, image=image, max_tokens=max_tokens, temperature=0.7, realtime_context=realtime_context)
        
        return response
    
    def run_interactive_session(self, duration=None, use_wake_word=True, greeting=True):
        """
        Run interactive Q&A session.
        
        Args:
            duration: Session duration in seconds (None = until interrupted)
            use_wake_word: If True, require wake word. If False, listen continuously.
            greeting: If True, say greeting message on startup
        """
        print("\n" + "="*60)
        print("🤖 Smart Assistant Active")
        print("="*60)
        if use_wake_word:
            print(f"Wake words: {', '.join(self.wake_words)}")
            print("Say a wake word, then ask your question!")
        else:
            print("Continuous listening mode - just ask your question!")
        print("Press Ctrl+C to exit")
        print("="*60 + "\n")
        
        # Say greeting on startup
        if greeting:
            # Wait a moment for audio system to be fully ready
            print("[Assistant] Waiting for audio system to be ready...")
            time.sleep(3)  # Give audio devices time to initialize (increased from 2 to 3 seconds)
            
            greeting_text = "Hi! I'm Rovy. Your assistant robot. How can I help you today? You can ask me any question, or give me a movement command"
            print(f"[Assistant] 🔊 Greeting: {greeting_text}")
            self.respeaker.set_led('speak')
            # Ensure audio device is ready before speaking
            self.tts._wait_for_audio_device(max_attempts=15, delay=0.3)
            self.tts.speak(greeting_text)
            self.respeaker.set_led('off')
            print("[Assistant] 👂 Now listening for questions...\n")
        
        start_time = time.time()
        
        try:
            while True:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    print("\n[Assistant] Session time limit reached")
                    break
                
                if use_wake_word:
                    # Listen for wake word
                    if self.listen_for_wake_word(timeout=5):
                        # Wake word detected - listen for question
                        self.respeaker.set_led('listen')
                        question = self.respeaker.listen(timeout=10, phrase_time_limit=15)
                        self.respeaker.set_led('off')
                    else:
                        continue
                else:
                    # No wake word - listen directly
                    print("[ReSpeaker] 🎤 Listening for question...")
                    self.respeaker.set_led('listen')
                    question = self.respeaker.listen(timeout=10, phrase_time_limit=15)
                    self.respeaker.set_led('off')
                
                if question:
                    # Start total timer
                    total_start = time.time()
                    
                    # Clean up question
                    question = question.strip()
                    
                    print(f"\n[Assistant] ❓ Question: {question}")
                    
                    # Check for movement commands FIRST (faster than LLM)
                    movement_cmd = self._check_movement_command(question)
                    if movement_cmd:
                        # LED: Thinking/Processing movement
                        self.respeaker.set_led('think')
                        response = self._execute_movement(movement_cmd)
                        total_time = time.time() - total_start
                        
                        print(f"[Assistant] 💬 Response: {response}")
                        print(f"[Assistant] ⏱️  Total time: {total_time:.2f}s")
                        # LED: Speaking response
                        self.respeaker.set_led('speak')
                        self.tts.speak(response)
                        self.respeaker.set_led('off')
                        continue
                    
                    # Check for vision-related keywords
                    vision_keywords = ['see', 'look', 'what', 'show', 'describe', 'view', 'image']
                    use_vision = any(kw in question.lower() for kw in vision_keywords)
                    
                    # Process question with LLM
                    self.respeaker.set_led('think')
                    
                    response = self.process_question(question, use_vision=use_vision)
                    
                    # Calculate total time
                    total_time = time.time() - total_start
                    
                    print(f"[Assistant] 💬 Answer: {response}")
                    print(f"[Assistant] ⏱️  Total time: {total_time:.1f}s")
                    
                    # Speak response
                    self.respeaker.set_led('speak')
                    self.tts.speak(response)
                    self.respeaker.set_led('off')
                else:
                    if not use_wake_word:
                        print("[Assistant] ⚠️  No question heard")
        
        except KeyboardInterrupt:
            print("\n[Assistant] Session ended by user")
        
        finally:
            self.respeaker.set_led('off')
            # Stop automatic navigation if running
            if self.auto_nav_process is not None:
                try:
                    self._stop_movement()
                except:
                    pass
    
    def run_continuous(self):
        """Run assistant continuously in background."""
        print("[Assistant] Starting continuous listening mode...")
        self.run_interactive_session(duration=None)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Assistant')
    parser.add_argument('--no-speakers', action='store_true', help='Disable audio output (print text only)')
    parser.add_argument('--duration', type=int, help='Session duration in seconds')
    parser.add_argument('--no-wake-word', action='store_true', help='Skip wake word, listen continuously')
    parser.add_argument('--no-greeting', action='store_true', help='Skip greeting message on startup')
    parser.add_argument('--port', default='/dev/ttyACM0', help='Rover serial port')
    parser.add_argument('--no-motors', action='store_true', help='Run without motor control')
    parser.add_argument('--weather-api-key', type=str, help='OpenWeatherMap API key for weather info (get free key at openweathermap.org)')
    
    args = parser.parse_args()
    
    # Initialize rover controller if not disabled
    motor_controller = None
    if not args.no_motors:
        try:
            from rover_controller import Rover
            print("[Assistant] Connecting to rover...")
            motor_controller = Rover(port=args.port)
            print("[Assistant] ✅ Rover connected")
        except Exception as e:
            print(f"[Assistant] ⚠️  Could not connect to rover: {e}")
            print("[Assistant] Running without motor control")
    
    # Test mode - run without camera
    print("\n" + "="*60)
    print("Smart Assistant" + (" - Motors Enabled" if motor_controller else " - No Motors"))
    print("="*60)
    
    assistant = SmartAssistant(
        camera=None,
        motor_controller=motor_controller,
        print_only=args.no_speakers,  # Speakers enabled by default, can disable with --no-speakers
        weather_api_key=args.weather_api_key  # Optional weather API key
    )
    
    try:
        assistant.run_interactive_session(
            duration=args.duration, 
            use_wake_word=not args.no_wake_word,
            greeting=not args.no_greeting
        )
    finally:
        # Cleanup
        if motor_controller:
            try:
                motor_controller.stop()
                motor_controller.cleanup()
            except:
                pass
        # Stop automatic navigation if running
        if assistant.auto_nav_process is not None:
            try:
                assistant._stop_movement()
            except:
                pass


"""
Speech Processing - Whisper STT and TTS
Uses local Whisper for speech recognition and espeak/Piper for TTS.
"""
import os
import re
import io
import wave
import tempfile
import subprocess
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger('Speech')

# Try to import Whisper
WHISPER_OK = False
try:
    import whisper
    WHISPER_OK = True
except ImportError:
    logger.warning("Whisper not available. Install: pip install openai-whisper")

# Try to import Piper
PIPER_OK = False
try:
    from piper import PiperVoice
    PIPER_OK = True
except ImportError:
    pass  # Will use espeak fallback


class SpeechProcessor:
    """Speech recognition and synthesis using local models."""
    
    def __init__(self, whisper_model: str = "base", tts_engine: str = "espeak"):
        self.whisper_model = None
        self.piper_voice = None
        self.tts_engine = tts_engine
        
        # Load Whisper
        if WHISPER_OK:
            try:
                logger.info(f"Loading Whisper ({whisper_model})...")
                self.whisper_model = whisper.load_model(whisper_model)
                logger.info("✅ Whisper ready")
            except Exception as e:
                logger.error(f"Whisper load failed: {e}")
        
        # Setup TTS
        if tts_engine == "piper" and PIPER_OK:
            self._init_piper()
        else:
            self._check_espeak()
    
    def _init_piper(self):
        """Initialize Piper TTS."""
        voice_paths = [
            os.path.expanduser("~/.local/share/piper-voices/en_US-lessac-medium.onnx"),
            os.path.expanduser("~/.local/share/piper-voices/en_US-hfc_male-medium.onnx"),
            "./voices/en_US-lessac-medium.onnx",
        ]
        
        for path in voice_paths:
            if os.path.exists(path):
                try:
                    self.piper_voice = PiperVoice.load(path)
                    logger.info(f"✅ Piper ready: {os.path.basename(path)}")
                    return
                except Exception as e:
                    logger.warning(f"Piper load failed: {e}")
        
        logger.info("No Piper voice found, using espeak")
        self._check_espeak()
    
    def _check_espeak(self):
        """Check if espeak is available."""
        try:
            result = subprocess.run(['espeak', '--version'], capture_output=True, timeout=2)
            if result.returncode == 0:
                self.tts_engine = "espeak"
                logger.info("✅ espeak ready")
        except:
            logger.warning("espeak not available")
            self.tts_engine = "none"
    
    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> Optional[str]:
        """Transcribe audio to text using Whisper."""
        if not self.whisper_model:
            logger.error("Whisper not loaded")
            return None
        
        try:
            # Convert bytes to float array
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                factor = 16000 / sample_rate
                new_len = int(len(audio) * factor)
                audio = np.interp(
                    np.linspace(0, len(audio), new_len),
                    np.arange(len(audio)),
                    audio
                )
            
            # Transcribe
            result = self.whisper_model.transcribe(
                audio,
                language="en",
                fp16=False,
                verbose=False
            )
            
            text = result["text"].strip()
            if text:
                logger.info(f"Transcribed: '{text}'")
            return text if text else None
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None
    
    def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio (WAV bytes)."""
        if not text:
            return None
        
        # Preprocess for TTS
        text = self._preprocess(text)
        logger.info(f"Synthesizing: '{text[:50]}...'")
        
        if self.piper_voice and PIPER_OK:
            return self._synth_piper(text)
        elif self.tts_engine == "espeak":
            return self._synth_espeak(text)
        else:
            logger.warning("No TTS engine available")
            return None
    
    def _synth_piper(self, text: str) -> Optional[bytes]:
        """Synthesize using Piper."""
        try:
            audio_data = []
            for chunk in self.piper_voice.synthesize_stream_raw(text):
                audio_data.append(chunk)
            
            if not audio_data:
                return None
            
            raw = b''.join(audio_data)
            
            # Convert to WAV
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(22050)
                wav.writeframes(raw)
            
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Piper synthesis failed: {e}")
            return None
    
    def _synth_espeak(self, text: str, speed: int = 150) -> Optional[bytes]:
        """Synthesize using espeak."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
            
            subprocess.run(
                ['espeak', '-w', temp_path, '-s', str(speed), text],
                capture_output=True,
                timeout=30
            )
            
            with open(temp_path, 'rb') as f:
                audio = f.read()
            
            os.unlink(temp_path)
            return audio
            
        except Exception as e:
            logger.error(f"espeak synthesis failed: {e}")
            return None
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text for better TTS pronunciation."""
        # Number to words (simple cases)
        def num_to_word(m):
            n = int(m.group(0))
            words = ['zero', 'one', 'two', 'three', 'four', 'five', 
                    'six', 'seven', 'eight', 'nine', 'ten']
            if n < len(words):
                return words[n]
            return m.group(0)
        
        text = re.sub(r'\b(\d)\b', num_to_word, text)
        
        # Abbreviations
        abbrevs = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missus',
            r'\bi\.e\.': 'that is',
            r'\be\.g\.': 'for example',
        }
        for pat, repl in abbrevs.items():
            text = re.sub(pat, repl, text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text


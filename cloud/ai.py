"""
Cloud AI Assistant - Using Local LLM Models
Uses llama.cpp with Gemma/Llama for text and LLaVA for vision.
"""
import os
import re
import gc
import time
import glob
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger('Assistant')

# Try to import llama-cpp-python
LLAMA_CPP_OK = False
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    LLAMA_CPP_OK = True
except ImportError:
    logger.warning("llama-cpp-python not available")
    logger.warning("Install: pip install llama-cpp-python")
    logger.warning("For GPU: CMAKE_ARGS=\"-DLLAMA_CUDA=on\" pip install llama-cpp-python")

try:
    from PIL import Image
    import numpy as np
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False


class CloudAssistant:
    """AI Assistant using local LLM models via llama.cpp"""
    
    def __init__(
        self,
        text_model_path: str = None,
        vision_model_path: str = None,
        vision_mmproj_path: str = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        lazy_load_vision: bool = True
    ):
        self.text_model_path = text_model_path
        self.vision_model_path = vision_model_path
        self.vision_mmproj_path = vision_mmproj_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        
        self.text_llm = None
        self.vision_llm = None
        self.vision_handler = None
        
        if not LLAMA_CPP_OK:
            logger.error("llama-cpp-python is required!")
            return
        
        # Find models
        self._find_models()
        
        logger.info(f"Text model: {self.text_model_path}")
        logger.info(f"Vision model: {self.vision_model_path}")
        
        # Clear GPU memory
        self._free_gpu()
        
        # Load text model
        self._load_text_model()
        
        # Load vision model (lazy or now)
        if not lazy_load_vision:
            self._load_vision_model()
    
    def _find_models(self):
        """Auto-detect model paths."""
        search_dirs = [
            os.path.expanduser("~/.cache"),
            os.path.expanduser("~/models"),
            "./models",
            "C:/models",
            "D:/models",
        ]
        
        # Find text model
        if not self.text_model_path:
            patterns = ["gemma*.gguf", "llama*.gguf", "mistral*.gguf", "phi*.gguf"]
            self.text_model_path = self._search_model(search_dirs, patterns)
        
        # Find vision model
        if not self.vision_model_path:
            patterns = ["llava*.gguf"]
            self.vision_model_path = self._search_model(search_dirs, patterns)
        
        # Find vision projector
        if not self.vision_mmproj_path and self.vision_model_path:
            patterns = ["mmproj*.gguf", "llava*mmproj*.gguf"]
            self.vision_mmproj_path = self._search_model(search_dirs, patterns)
    
    def _search_model(self, dirs: list, patterns: list) -> Optional[str]:
        """Search for a model file."""
        for directory in dirs:
            if not os.path.exists(directory):
                continue
            for pattern in patterns:
                matches = glob.glob(os.path.join(directory, pattern))
                for match in matches:
                    if os.path.isfile(match) and os.path.getsize(match) > 1_000_000:
                        return match
        return None
    
    def _free_gpu(self):
        """Free GPU memory."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def _load_text_model(self):
        """Load the text LLM."""
        if self.text_llm:
            return
        
        if not self.text_model_path or not os.path.exists(self.text_model_path):
            logger.error(f"Text model not found: {self.text_model_path}")
            return
        
        logger.info("Loading text model...")
        start = time.time()
        
        try:
            self.text_llm = Llama(
                model_path=self.text_model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=os.cpu_count() or 4,
                n_batch=512,
                verbose=False,
            )
            logger.info(f"✅ Text model loaded in {time.time() - start:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
    
    def _load_vision_model(self):
        """Load the vision LLM."""
        if self.vision_llm:
            return
        
        if not self.vision_model_path or not os.path.exists(self.vision_model_path):
            logger.warning(f"Vision model not found: {self.vision_model_path}")
            return
        
        if not self.vision_mmproj_path or not os.path.exists(self.vision_mmproj_path):
            logger.warning(f"Vision projector not found: {self.vision_mmproj_path}")
            return
        
        logger.info("Loading vision model...")
        start = time.time()
        
        try:
            self.vision_handler = Llava15ChatHandler(
                clip_model_path=self.vision_mmproj_path,
                verbose=False
            )
            
            self.vision_llm = Llama(
                model_path=self.vision_model_path,
                chat_handler=self.vision_handler,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=2048,
                n_threads=os.cpu_count() or 4,
                n_batch=512,
                logits_all=True,
                verbose=False,
            )
            logger.info(f"✅ Vision model loaded in {time.time() - start:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
    
    def ask(self, question: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        """Ask a text-only question."""
        if not self.text_llm:
            self._load_text_model()
            if not self.text_llm:
                return "Text model not available."
        
        logger.info(f"Query: {question}")
        start = time.time()
        
        try:
            # Add time context if needed
            time_ctx = ""
            if any(p in question.lower() for p in ['time', 'date', 'today', 'day']):
                time_ctx = f"Current: {datetime.now().strftime('%I:%M %p, %A %B %d, %Y')}. "
            
            response = self.text_llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": f"You are Rovy, a helpful robot assistant. {time_ctx}Give concise answers under 50 words."
                    },
                    {"role": "user", "content": question}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            answer = response['choices'][0]['message']['content'].strip()
            answer = self._clean(answer)
            
            logger.info(f"Response in {time.time() - start:.1f}s")
            return answer
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Error: {e}"
    
    def ask_with_vision(self, question: str, image, max_tokens: int = 200) -> str:
        """Ask a question about an image."""
        if not self.vision_llm:
            self._load_vision_model()
            if not self.vision_llm:
                return "Vision model not available."
        
        logger.info(f"Vision query: {question}")
        start = time.time()
        
        self.vision_llm.reset()
        
        try:
            # Convert to PIL Image
            pil_img = self._to_pil(image)
            if not pil_img:
                return "Could not process image."
            
            # Convert to base64
            import io
            import base64
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=70)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            
            response = self.vision_llm.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": question}
                    ]
                }],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            
            answer = ""
            if 'choices' in response and response['choices']:
                choice = response['choices'][0]
                if 'message' in choice:
                    answer = choice['message'].get('content', '')
            
            answer = self._clean(answer.strip())
            logger.info(f"Vision response in {time.time() - start:.1f}s")
            return answer if answer else "I couldn't understand what I'm seeing."
            
        except Exception as e:
            logger.error(f"Vision query failed: {e}")
            return f"Error: {e}"
    
    def _to_pil(self, image):
        """Convert various formats to PIL Image."""
        if not PIL_OK:
            return None
        
        try:
            if isinstance(image, Image.Image):
                return image
            if isinstance(image, bytes):
                import io
                return Image.open(io.BytesIO(image))
            if isinstance(image, np.ndarray):
                if CV2_OK:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return Image.fromarray(image)
            if isinstance(image, str) and os.path.exists(image):
                return Image.open(image)
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
        return None
    
    def _clean(self, text: str) -> str:
        """Clean model output for TTS."""
        text = text.replace('</s>', '').replace('#', '')
        text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)  # Remove emojis
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Remove italic
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_movement(self, response: str, query: str) -> Optional[Dict[str, Any]]:
        """Extract movement commands from text."""
        text = f"{query} {response}".lower()
        
        patterns = {
            'forward': [r'go\s+forward', r'move\s+forward', r'ahead'],
            'backward': [r'go\s+back', r'move\s+back', r'reverse'],
            'left': [r'turn\s+left', r'go\s+left'],
            'right': [r'turn\s+right', r'go\s+right'],
            'stop': [r'\bstop\b', r'halt']
        }
        
        for direction, pats in patterns.items():
            for pat in pats:
                if re.search(pat, text):
                    dist = 0.5
                    if 'little' in text:
                        dist = 0.2
                    elif 'far' in text or 'lot' in text:
                        dist = 1.0
                    
                    speed = 'medium'
                    if 'slow' in text:
                        speed = 'slow'
                    elif 'fast' in text:
                        speed = 'fast'
                    
                    return {'direction': direction, 'distance': dist, 'speed': speed}
        
        return None


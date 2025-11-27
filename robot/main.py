#!/usr/bin/env python3
"""
Rovy Raspberry Pi Client
Connects to rover hardware and streams to PC server via Tailscale.

Usage:
    python main.py
"""
import asyncio
import json
import time
import base64
import signal
import sys
import threading
from datetime import datetime

import config

# Optional imports with fallbacks
try:
    import websockets
    WEBSOCKETS_OK = True
except ImportError:
    WEBSOCKETS_OK = False
    print("ERROR: websockets not installed. Run: pip install websockets")

try:
    import cv2
    CAMERA_OK = True
except ImportError:
    CAMERA_OK = False
    print("WARNING: OpenCV not installed. Camera disabled.")

try:
    import pyaudio
    import numpy as np
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False
    print("WARNING: PyAudio not installed. Microphone disabled.")

try:
    import sounddevice as sd
    import soundfile as sf
    import io
    PLAYBACK_OK = True
except ImportError:
    PLAYBACK_OK = False
    print("WARNING: sounddevice not installed. Audio playback disabled.")

try:
    from rover import Rover
    ROVER_OK = True
except Exception as e:
    ROVER_OK = False
    print(f"WARNING: Rover not available: {e}")


class RovyClient:
    """
    Client that runs on Raspberry Pi.
    - Connects to rover via serial
    - Streams audio/video to PC server
    - Receives commands from server
    """
    
    def __init__(self):
        self.running = False
        self.ws = None
        self.rover = None
        self.camera = None
        self.audio_stream = None
        
        # State
        self.is_listening = False
        self.audio_buffer = []
        self.last_image = None
        
        print("=" * 50)
        print("  ROVY RASPBERRY PI CLIENT")
        print(f"  Server: {config.SERVER_URL}")
        print("=" * 50)
    
    def init_rover(self):
        """Initialize rover connection."""
        if not ROVER_OK:
            print("[Rover] Not available")
            return False
        
        try:
            self.rover = Rover(config.ROVER_SERIAL_PORT, config.ROVER_BAUDRATE)
            self.rover.display_lines([
                "ROVY Cloud",
                "Connecting...",
                config.PC_SERVER_IP,
                ""
            ])
            print("[Rover] Connected")
            return True
        except Exception as e:
            print(f"[Rover] Connection failed: {e}")
            return False
    
    def init_camera(self):
        """Initialize camera."""
        if not CAMERA_OK:
            return False
        
        try:
            self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            ret, _ = self.camera.read()
            if ret:
                print("[Camera] Ready")
                return True
            else:
                print("[Camera] Failed to read frame")
                return False
        except Exception as e:
            print(f"[Camera] Init failed: {e}")
            return False
    
    def init_audio(self):
        """Initialize audio input (ReSpeaker)."""
        if not AUDIO_OK:
            return False
        
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # Find ReSpeaker device
            device_index = None
            for i in range(self.pyaudio.get_device_count()):
                info = self.pyaudio.get_device_info_by_index(i)
                name = info.get('name', '').lower()
                if 'respeaker' in name or 'seeed' in name:
                    device_index = i
                    print(f"[Audio] Found ReSpeaker: {info['name']}")
                    break
            
            self.audio_device_index = device_index
            print("[Audio] Ready")
            return True
        except Exception as e:
            print(f"[Audio] Init failed: {e}")
            return False
    
    async def connect_server(self):
        """Connect to PC server via WebSocket."""
        if not WEBSOCKETS_OK:
            return False
        
        attempt = 0
        while self.running:
            attempt += 1
            try:
                print(f"[Server] Connecting to {config.SERVER_URL} (attempt {attempt})...")
                
                self.ws = await websockets.connect(
                    config.SERVER_URL,
                    ping_interval=30,
                    ping_timeout=10
                )
                
                print("[Server] Connected!")
                
                if self.rover:
                    self.rover.display_lines([
                        "ROVY Cloud",
                        "Connected!",
                        config.PC_SERVER_IP,
                        datetime.now().strftime("%H:%M:%S")
                    ])
                
                return True
                
            except Exception as e:
                print(f"[Server] Connection failed: {e}")
                
                if self.rover:
                    self.rover.display_lines([
                        "ROVY Cloud",
                        "Reconnecting...",
                        f"Attempt {attempt}",
                        str(e)[:21]
                    ])
                
                if config.MAX_RECONNECT_ATTEMPTS > 0 and attempt >= config.MAX_RECONNECT_ATTEMPTS:
                    print("[Server] Max reconnect attempts reached")
                    return False
                
                await asyncio.sleep(config.RECONNECT_DELAY)
        
        return False
    
    async def send_message(self, msg_type: str, **kwargs):
        """Send a message to the server."""
        if not self.ws:
            return
        
        message = {
            "type": msg_type,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        
        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            print(f"[Send] Error: {e}")
    
    async def handle_message(self, raw_message: str):
        """Handle incoming message from server."""
        try:
            msg = json.loads(raw_message)
            msg_type = msg.get('type', '')
            
            if msg_type == 'speak':
                await self.handle_speak(msg)
            
            elif msg_type == 'move':
                await self.handle_move(msg)
            
            elif msg_type == 'gimbal':
                await self.handle_gimbal(msg)
            
            elif msg_type == 'lights':
                await self.handle_lights(msg)
            
            elif msg_type == 'display':
                await self.handle_display(msg)
            
            elif msg_type == 'pong':
                pass  # Heartbeat response
            
            elif msg_type == 'error':
                print(f"[Server Error] {msg.get('error', 'Unknown error')}")
            
            else:
                print(f"[Unknown message type] {msg_type}")
                
        except json.JSONDecodeError as e:
            print(f"[Parse Error] {e}")
    
    async def handle_speak(self, msg):
        """Handle speak command - play TTS audio."""
        text = msg.get('text', '')
        audio_b64 = msg.get('audio_base64')
        
        print(f"[Speak] {text[:50]}...")
        
        if audio_b64 and PLAYBACK_OK:
            try:
                audio_bytes = base64.b64decode(audio_b64)
                
                # Play WAV audio
                audio_io = io.BytesIO(audio_bytes)
                data, samplerate = sf.read(audio_io)
                sd.play(data, samplerate)
                sd.wait()
                
            except Exception as e:
                print(f"[Speak] Playback error: {e}")
        else:
            # Fallback: use espeak locally
            import subprocess
            try:
                subprocess.run(['espeak', text], timeout=30)
            except:
                print(f"[Speak] No playback available: {text}")
    
    async def handle_move(self, msg):
        """Handle movement command."""
        if not self.rover:
            return
        
        direction = msg.get('direction', 'stop')
        distance = msg.get('distance', 0.5)
        speed = msg.get('speed', 'medium')
        
        print(f"[Move] {direction} {distance}m at {speed}")
        
        # Run movement in thread to not block
        def do_move():
            self.rover.move(direction, distance, speed)
        
        threading.Thread(target=do_move, daemon=True).start()
    
    async def handle_gimbal(self, msg):
        """Handle gimbal command."""
        if not self.rover:
            return
        
        action = msg.get('action', 'move')
        pan = msg.get('pan', 0)
        tilt = msg.get('tilt', 0)
        speed = msg.get('speed', 200)
        
        print(f"[Gimbal] {action} pan={pan} tilt={tilt}")
        
        if action == 'nod':
            threading.Thread(target=self.rover.nod_yes, daemon=True).start()
        elif action == 'shake':
            threading.Thread(target=self.rover.shake_no, daemon=True).start()
        elif action == 'reset':
            self.rover.gimbal_ctrl(0, 0, speed, 10)
        else:
            self.rover.gimbal_ctrl(pan, tilt, speed, 10)
    
    async def handle_lights(self, msg):
        """Handle lights command."""
        if not self.rover:
            return
        
        front = msg.get('front', 0)
        back = msg.get('back', 0)
        self.rover.lights_ctrl(front, back)
    
    async def handle_display(self, msg):
        """Handle OLED display command."""
        if not self.rover:
            return
        
        lines = msg.get('lines', [])
        self.rover.display_lines(lines)
    
    def capture_image(self) -> bytes:
        """Capture image from camera as JPEG bytes."""
        if not self.camera:
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        return buffer.tobytes()
    
    def record_audio(self, duration: float) -> bytes:
        """Record audio from microphone."""
        if not AUDIO_OK or not self.pyaudio:
            return None
        
        try:
            frames = []
            
            stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=config.CHANNELS,
                rate=config.SAMPLE_RATE,
                input=True,
                input_device_index=self.audio_device_index,
                frames_per_buffer=config.CHUNK_SIZE
            )
            
            num_chunks = int(config.SAMPLE_RATE / config.CHUNK_SIZE * duration)
            
            for _ in range(num_chunks):
                data = stream.read(config.CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            return b''.join(frames)
            
        except Exception as e:
            print(f"[Audio] Record error: {e}")
            return None
    
    async def stream_loop(self):
        """Main loop for streaming audio/video to server."""
        print("[Stream] Starting...")
        
        image_interval = 1.0 / config.CAMERA_FPS
        audio_interval = config.AUDIO_BUFFER_SECONDS
        sensor_interval = 5.0
        
        last_image_time = 0
        last_audio_time = 0
        last_sensor_time = 0
        
        while self.running and self.ws:
            try:
                now = time.time()
                
                # Send image periodically
                if CAMERA_OK and self.camera and (now - last_image_time) >= image_interval:
                    image_bytes = self.capture_image()
                    if image_bytes:
                        self.last_image = image_bytes
                        await self.send_message(
                            "image_data",
                            image_base64=base64.b64encode(image_bytes).decode('utf-8'),
                            width=config.CAMERA_WIDTH,
                            height=config.CAMERA_HEIGHT
                        )
                    last_image_time = now
                
                # Send audio periodically
                if AUDIO_OK and self.is_listening and (now - last_audio_time) >= audio_interval:
                    audio_bytes = self.record_audio(audio_interval)
                    if audio_bytes:
                        await self.send_message(
                            "audio_data",
                            audio_base64=base64.b64encode(audio_bytes).decode('utf-8'),
                            sample_rate=config.SAMPLE_RATE,
                            duration=audio_interval
                        )
                    last_audio_time = now
                
                # Send sensor data periodically
                if self.rover and (now - last_sensor_time) >= sensor_interval:
                    status = self.rover.get_status()
                    if status:
                        await self.send_message(
                            "sensor_data",
                            battery_voltage=status.get('voltage'),
                            battery_percent=self.rover.voltage_to_percent(status.get('voltage')),
                            temperature=status.get('temperature'),
                            imu_roll=status.get('roll'),
                            imu_pitch=status.get('pitch'),
                            imu_yaw=status.get('yaw')
                        )
                    last_sensor_time = now
                
                await asyncio.sleep(0.01)
                
            except websockets.exceptions.ConnectionClosed:
                print("[Stream] Connection lost")
                break
            except Exception as e:
                print(f"[Stream] Error: {e}")
                await asyncio.sleep(0.1)
    
    async def receive_loop(self):
        """Receive messages from server."""
        print("[Receive] Starting...")
        
        while self.running and self.ws:
            try:
                message = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                await self.handle_message(message)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("[Receive] Connection lost")
                break
            except Exception as e:
                print(f"[Receive] Error: {e}")
    
    async def run(self):
        """Main run loop."""
        self.running = True
        
        # Initialize hardware
        self.init_rover()
        self.init_camera()
        self.init_audio()
        
        # Main loop with reconnection
        while self.running:
            if await self.connect_server():
                self.is_listening = True
                
                # Run stream and receive loops
                stream_task = asyncio.create_task(self.stream_loop())
                receive_task = asyncio.create_task(self.receive_loop())
                
                # Wait for either to finish (connection lost)
                done, pending = await asyncio.wait(
                    [stream_task, receive_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                
                self.is_listening = False
                
                if self.ws:
                    await self.ws.close()
                    self.ws = None
                
                if self.running:
                    print("[Main] Reconnecting...")
                    await asyncio.sleep(config.RECONNECT_DELAY)
            else:
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        if self.camera:
            self.camera.release()
        
        if hasattr(self, 'pyaudio') and self.pyaudio:
            self.pyaudio.terminate()
        
        if self.rover:
            self.rover.display_lines(["ROVY", "Disconnected", "", ""])
            self.rover.cleanup()
        
        print("[Client] Cleanup complete")
    
    def stop(self):
        """Stop the client."""
        print("[Client] Stopping...")
        self.running = False


# Signal handler for graceful shutdown
client = None

def signal_handler(sig, frame):
    print("\n[Signal] Shutting down...")
    if client:
        client.stop()
    sys.exit(0)


def main():
    global client
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    client = RovyClient()
    
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        client.stop()


if __name__ == "__main__":
    main()


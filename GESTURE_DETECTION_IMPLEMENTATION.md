# Gesture Detection Implementation

## Overview
Added gesture detection to the voice control screen that allows the robot eyes to react to hand gestures detected by the front camera. The system detects gestures like "like" (thumbs up) and "heart" shapes, and updates the robot's emotional expression accordingly.

## Implementation Details

### Server-Side (Cloud)

1. **Gesture Detection Service** (`cloud/app/gesture_detection.py`)
   - Uses MediaPipe Hands for hand landmark detection
   - Detects gestures: "like" (thumbs up) and "heart" shape
   - Returns gesture type and confidence score

2. **API Endpoint** (`cloud/app/main.py`)
   - `POST /gesture/detect` - Accepts image upload and returns detected gesture
   - Handles image decoding and gesture classification
   - Returns JSON: `{gesture: 'like'|'heart'|'none', confidence: float}`

3. **Dependencies Added**
   - `mediapipe>=0.10.0` - Hand pose detection
   - `opencv-python>=4.8.0` - Image processing

### Client-Side (Mobile App)

1. **Gesture Detection Service** (`mobile/services/gesture-detection.ts`)
   - Communicates with server API
   - Converts image data URIs to uploadable format
   - Throttles requests to avoid overwhelming server

2. **Gesture Detection Hook** (`mobile/hooks/useGestureDetection.ts`)
   - React hook for gesture detection
   - Manages detection state and callbacks
   - Provides `detectGestureFromImage()` function

3. **Gesture Detection Camera Component** (`mobile/components/gesture-detection-camera.tsx`)
   - Hidden front camera component
   - Captures frames periodically (every 500ms)
   - Sends frames to server for gesture detection
   - Runs in background, invisible to user

4. **Voice Control Screen Integration** (`mobile/app/agentic.tsx`)
   - Integrated gesture detection camera
   - Updates robot eyes emotion based on detected gestures:
     - "like" gesture → `happy` emotion
     - "heart" gesture → `love` emotion (heart eyes)
   - Gesture emotions override state-based emotions
   - Gesture emotions reset after 2 seconds

### Dependencies Added

**Mobile (`mobile/package.json`):**
- `expo-camera`: `~16.0.9` - Camera access
- `expo-image-manipulator`: `~13.0.1` - Image processing

**Cloud (`cloud/requirements.txt`):**
- `mediapipe>=0.10.0` - Hand gesture detection
- `opencv-python>=4.8.0` - Image processing

### Permissions

**Mobile (`mobile/app.json`):**
- Added iOS camera permission description
- Added expo-camera plugin configuration

## How It Works

1. **User opens Voice Control screen**
   - Front camera starts in background (hidden)
   - Camera permission is requested if not already granted

2. **Frame Capture**
   - Camera captures frames every 500ms
   - Frames are converted to JPEG (low quality for speed)
   - Sent to server via POST request

3. **Server Processing**
   - MediaPipe Hands analyzes hand landmarks
   - Gesture classification:
     - Thumbs up detected → "like"
     - Heart shape detected → "heart"
   - Returns gesture and confidence score

4. **Emotion Update**
   - Client receives gesture result
   - Robot eyes emotion updates:
     - "like" → Happy eyes
     - "heart" → Heart eyes (love emotion)
   - Emotion persists for 2 seconds, then resets

## Gesture Recognition

### "Like" Gesture (Thumbs Up)
- Thumb extended upward
- Other fingers closed
- Confidence threshold: 0.7

### "Heart" Gesture
- Index and middle fingers curved together
- Tips close together forming a point
- Ring finger also curved (optional)
- Confidence threshold: 0.6

## Usage

The gesture detection runs automatically when:
- Voice Control screen is focused
- Audio connection is active
- Camera permission is granted

No user action required - just show gestures to the front camera and the robot eyes will react!

## Installation

### Server
```bash
cd cloud
pip install -r requirements.txt
```

### Mobile
```bash
cd mobile
npm install
```

Then rebuild the app:
```bash
npm run ios  # or npm run android
```

## Future Enhancements

- More gesture types (peace sign, OK sign, etc.)
- Emotion detection from facial expressions
- Real-time gesture visualization overlay
- Gesture-based robot commands
- Improved gesture recognition accuracy


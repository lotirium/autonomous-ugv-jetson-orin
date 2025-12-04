# Multilingual Push-to-Talk Support

## Overview

The Rovy robot now supports **automatic language detection** for push-to-talk interactions. When a user speaks in any language, the system will:

1. **Detect** the language automatically using Whisper
2. **Transcribe** the speech in that language
3. **Respond** with text-to-speech in the **same language**

## Supported Languages

### Speech Recognition (Whisper)
Whisper supports 90+ languages including:
- **English** (en)
- **Spanish** (es)
- **French** (fr)
- **German** (de)
- **Italian** (it)
- **Portuguese** (pt)
- **Russian** (ru)
- **Chinese** (zh)
- **Japanese** (ja)
- **Korean** (ko)
- **Arabic** (ar)
- **Hindi** (hi)
- **Dutch** (nl)
- **Polish** (pl)
- **Turkish** (tr)
- And many more...

### Text-to-Speech (espeak)
The system uses espeak for TTS, which supports 100+ languages with the same language codes as Whisper.

## How It Works

### 1. Audio Capture
When the user presses and holds the push-to-talk button, audio is recorded from the microphone.

### 2. Language Detection
```python
# In cloud/speech.py
result = whisper_model.transcribe(
    audio,
    language=None,  # Auto-detect language
    fp16=False,
    verbose=False
)

text = result["text"]
detected_language = result.get("language", "en")
```

### 3. Response Generation
The AI assistant processes the query and generates a response (in the same language if the LLM supports it).

### 4. Speech Synthesis
```python
# In cloud/speech.py
audio_bytes = speech.synthesize(text, language=detected_language)
```

The TTS engine uses the detected language code to select the appropriate voice.

## Implementation Details

### Modified Files

#### 1. `cloud/speech.py`
- **`transcribe()`** now returns a dict with `text` and `language` keys
- **`synthesize()`** accepts a `language` parameter
- **`_synth_espeak()`** maps language codes to espeak voices

#### 2. `cloud/main.py` (WebSocket Server)
- **`RobotConnection`** tracks `last_language` for each session
- **`handle_audio()`** extracts language from transcription result
- **`process_query()`** accepts and passes through language parameter
- **`send_speak()`** uses detected language for TTS

#### 3. `cloud/app/main.py` (REST API)
- **`/stt` endpoint** returns language in response
- **`/tts` endpoint** accepts language in request
- **WebSocket audio handler** tracks language and passes to TTS

#### 4. `robot/main_api.py` (Robot API)
- **`/speak` endpoint** accepts language parameter from cloud

## API Examples

### Speech-to-Text (STT)
```bash
POST /stt
Content-Type: multipart/form-data

# Response:
{
  "text": "¿Cómo estás?",
  "language": "es",
  "success": true
}
```

### Text-to-Speech (TTS)
```bash
POST /tts
Content-Type: application/json

{
  "text": "Bonjour, comment allez-vous?",
  "language": "fr"
}

# Returns: audio/wav file with French speech
```

### WebSocket Audio Stream
```json
// Client sends:
{
  "type": "audio_data",
  "audio_base64": "...",
  "sample_rate": 16000
}

// Server responds:
{
  "type": "transcript",
  "text": "こんにちは",
  "language": "ja"
}

// Then server sends TTS in Japanese:
{
  "type": "speak",
  "text": "こんにちは、元気ですか？",
  "audio_base64": "..."
}
```

## Configuration

### Whisper Model Selection
In `cloud/config.py`:
```python
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
```

Larger models provide better accuracy for non-English languages:
- **tiny**: Fast, less accurate
- **base**: Good balance (default)
- **small**: Better accuracy
- **medium**: Very good for multilingual
- **large**: Best accuracy, slower

### TTS Engine
In `cloud/config.py`:
```python
TTS_ENGINE = "espeak"  # Default, supports 100+ languages
```

For better quality in specific languages, you can use language-specific Piper voices:
```python
# Download Piper voices for specific languages
# Example: Spanish voice
PIPER_VOICE_ES = "~/.local/share/piper-voices/es_ES-davefx-medium.onnx"
```

## Usage Examples

### Example 1: Spanish Conversation
```
User (speaking Spanish): "¿Qué ves frente a ti?"
Robot detects: language="es"
Robot transcribes: "¿Qué ves frente a ti?"
AI responds: "Veo una puerta y una mesa."
Robot speaks (in Spanish): "Veo una puerta y una mesa."
```

### Example 2: French Query
```
User (speaking French): "Avance de deux mètres"
Robot detects: language="fr"
Robot transcribes: "Avance de deux mètres"
AI responds: "D'accord, j'avance."
Robot speaks (in French): "D'accord, j'avance."
Robot moves forward 2 meters
```

### Example 3: Mixed Languages
```
User (speaking Japanese): "こんにちは"
Robot detects: language="ja"
Robot responds in Japanese

User (switching to English): "What do you see?"
Robot detects: language="en"
Robot responds in English
```

## Limitations

1. **AI Response Language**: The LLM (Qwen2-VL) may respond in English even if asked in another language. For full multilingual support, you may need to:
   - Use a multilingual LLM
   - Add a translation layer
   - Prompt the LLM to respond in the detected language

2. **TTS Quality**: espeak voices are robotic. For better quality:
   - Use Piper with language-specific voices
   - Use cloud TTS services (Google, Azure, etc.)

3. **Language Mixing**: If the user mixes languages in one utterance, Whisper will detect the dominant language.

## Future Enhancements

- [ ] Add Piper multilingual voice support
- [ ] Implement language preference persistence
- [ ] Add translation layer for LLM responses
- [ ] Support language switching commands ("Switch to Spanish")
- [ ] Add confidence scores for language detection
- [ ] Implement fallback to English if language not supported

## Testing

To test multilingual support:

1. Start the cloud server:
```bash
cd cloud
python main.py
```

2. Start the robot server:
```bash
cd robot
python main_api.py
```

3. Use the mobile app or test with curl:
```bash
# Test Spanish STT
curl -X POST http://localhost:8000/stt \
  -F "audio=@spanish_audio.wav"

# Test French TTS
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour", "language": "fr"}' \
  --output french_speech.wav
```

## Troubleshooting

### Language Not Detected Correctly
- Use a larger Whisper model (`small` or `medium`)
- Ensure audio quality is good (16kHz, clear speech)
- Speak clearly and avoid background noise

### TTS Not Working for Language
- Check if espeak supports the language: `espeak --voices`
- Verify language code is correct (ISO 639-1 format)
- Check logs for TTS errors

### Robot Responds in Wrong Language
- Check that `last_language` is being tracked correctly
- Verify language is passed through all API calls
- Ensure the LLM is prompted to respond in the detected language

## References

- [Whisper Language Support](https://github.com/openai/whisper#available-models-and-languages)
- [espeak Language Codes](http://espeak.sourceforge.net/languages.html)
- [Piper Voices](https://github.com/rhasspy/piper/blob/master/VOICES.md)


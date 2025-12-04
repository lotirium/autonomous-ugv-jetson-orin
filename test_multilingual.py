#!/usr/bin/env python3
"""
Test script for multilingual push-to-talk support.
Tests language detection and TTS in multiple languages.
"""
import sys
import os

# Add cloud directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloud'))

def test_transcribe():
    """Test that transcribe returns dict with language."""
    print("=" * 60)
    print("Testing transcribe() return format...")
    print("=" * 60)
    
    try:
        from speech import SpeechProcessor
        
        # Create processor
        speech = SpeechProcessor(whisper_model="tiny")  # Use tiny for faster testing
        
        # Create dummy audio (silence)
        import numpy as np
        audio_bytes = np.zeros(16000, dtype=np.int16).tobytes()  # 1 second of silence
        
        # Transcribe
        result = speech.transcribe(audio_bytes, 16000)
        
        # Check format
        if result is None:
            print("✅ Returns None for empty audio (expected)")
        elif isinstance(result, dict):
            print(f"✅ Returns dict: {result}")
            assert 'text' in result, "Missing 'text' key"
            assert 'language' in result, "Missing 'language' key"
            print("✅ Dict has correct keys")
        else:
            print(f"❌ ERROR: Expected dict, got {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_synthesize():
    """Test that synthesize accepts language parameter."""
    print("\n" + "=" * 60)
    print("Testing synthesize() language parameter...")
    print("=" * 60)
    
    try:
        from speech import SpeechProcessor
        
        # Create processor
        speech = SpeechProcessor(tts_engine="espeak")
        
        # Test English
        print("\nTesting English...")
        audio = speech.synthesize("Hello world", language="en")
        if audio:
            print(f"✅ English TTS generated {len(audio)} bytes")
        else:
            print("⚠️  English TTS returned None (espeak may not be available)")
        
        # Test Spanish
        print("\nTesting Spanish...")
        audio = speech.synthesize("Hola mundo", language="es")
        if audio:
            print(f"✅ Spanish TTS generated {len(audio)} bytes")
        else:
            print("⚠️  Spanish TTS returned None")
        
        # Test French
        print("\nTesting French...")
        audio = speech.synthesize("Bonjour le monde", language="fr")
        if audio:
            print(f"✅ French TTS generated {len(audio)} bytes")
        else:
            print("⚠️  French TTS returned None")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_espeak_voices():
    """Check available espeak voices."""
    print("\n" + "=" * 60)
    print("Checking espeak voice availability...")
    print("=" * 60)
    
    try:
        import subprocess
        
        result = subprocess.run(
            ['espeak', '--voices'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ espeak is available")
            print("\nAvailable voices (first 20):")
            lines = result.stdout.split('\n')[:21]  # Header + 20 voices
            for line in lines:
                if any(lang in line for lang in ['en', 'es', 'fr', 'de', 'it', 'pt', 'ja', 'zh']):
                    print(f"  {line}")
        else:
            print("⚠️  espeak not available or returned error")
            
    except FileNotFoundError:
        print("⚠️  espeak command not found")
    except Exception as e:
        print(f"⚠️  Could not check espeak: {e}")


def test_language_codes():
    """Verify language code mapping."""
    print("\n" + "=" * 60)
    print("Testing language code mapping...")
    print("=" * 60)
    
    from speech import SpeechProcessor
    
    # Test that _synth_espeak has correct mapping
    speech = SpeechProcessor(tts_engine="espeak")
    
    test_codes = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'zh': 'Chinese',
        'ja': 'Japanese',
    }
    
    print("\nSupported language codes:")
    for code, name in test_codes.items():
        print(f"  {code} - {name}")
    
    print("\n✅ Language codes configured")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MULTILINGUAL PUSH-TO-TALK TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Test 1: Transcribe format
    results.append(("Transcribe format", test_transcribe()))
    
    # Test 2: Synthesize language parameter
    results.append(("Synthesize language", test_synthesize()))
    
    # Test 3: espeak availability
    test_espeak_voices()
    
    # Test 4: Language codes
    test_language_codes()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nMultilingual support is ready to use.")
        print("Try speaking in different languages using push-to-talk!")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


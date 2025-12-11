#!/usr/bin/env python3
"""
Test script for "Come to Me" feature
Run this on the Raspberry Pi to test the complete flow
"""

import sys
import os

# Add oakd_navigation to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'oakd_navigation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robot'))

from rovy_integration import RovyNavigator

def test_person_detection_only():
    """Test just the person detection without navigation"""
    print("=" * 60)
    print("TEST 1: PERSON DETECTION ONLY")
    print("=" * 60)
    print("\nThis will test if the robot can detect you.")
    print("Stand 2-3 meters in front of the robot.")
    input("Press ENTER when ready...")
    
    from person_detector import PersonDetector
    
    detector = PersonDetector(confidence_threshold=0.5)
    
    try:
        detector.start()
        print("\n✅ Camera started")
        
        person = detector.scan_for_person(duration=5.0, min_detections=3)
        
        if person:
            print(f"\n✅ SUCCESS! Person detected:")
            print(f"   Distance: {person.distance:.2f}m")
            print(f"   Angle: {person.angle_horizontal:.1f}°")
            print(f"   Confidence: {person.confidence:.2%}")
            return True
        else:
            print("\n❌ FAILED: No person detected")
            print("   - Make sure you're in front of the camera")
            print("   - Check lighting (avoid backlighting)")
            print("   - Stand still during detection")
            return False
            
    finally:
        detector.stop()


def test_come_to_me_full():
    """Test the complete come to me feature"""
    print("\n" + "=" * 60)
    print("TEST 2: FULL 'COME TO ME' NAVIGATION")
    print("=" * 60)
    print("\nThis will make the robot come to you!")
    print("Make sure:")
    print("  1. Robot is on the ground (not on a table)")
    print("  2. There's clear space between you and robot (2-3m)")
    print("  3. No obstacles in the path")
    print("  4. You're ready to grab it if needed")
    input("\nPress ENTER to start...")
    
    navigator = None
    
    try:
        # Create navigator
        print("\n1. Initializing navigator...")
        navigator = RovyNavigator(rover_port='/dev/ttyAMA0')
        navigator.start()
        print("   ✅ Navigator started")
        
        # Run come to me
        print("\n2. Starting 'come to me' command...")
        print("   Stand in front of the robot now!")
        success = navigator.come_to_person(max_approach_distance=0.8)
        
        if success:
            print("\n✅ SUCCESS! Robot came to you!")
            print("   The robot should now be about 0.8m (2.6 feet) away")
            return True
        else:
            print("\n❌ FAILED: Robot couldn't come to you")
            print("   Check console output above for errors")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if navigator:
            print("\n3. Cleaning up...")
            navigator.cleanup()
            print("   ✅ Cleanup complete")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ROVY 'COME TO ME' FEATURE TEST")
    print("=" * 60)
    print("\nThis script will test:")
    print("  1. Person detection with OAK-D camera")
    print("  2. Full 'come to me' navigation")
    print("\nPress Ctrl+C at any time to abort")
    print("=" * 60)
    
    # Test 1: Person detection
    detection_ok = test_person_detection_only()
    
    if not detection_ok:
        print("\n⚠️  Person detection failed. Fix this before testing navigation.")
        print("\nTroubleshooting tips:")
        print("  - Ensure OAK-D camera is connected")
        print("  - Check USB connection")
        print("  - Verify you're in camera view")
        print("  - Try: python oakd_navigation/person_detector.py")
        return
    
    # Test 2: Full navigation
    print("\n" + "=" * 60)
    nav_ok = test_come_to_me_full()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Person Detection: {'✅ PASS' if detection_ok else '❌ FAIL'}")
    print(f"Come to Me:       {'✅ PASS' if nav_ok else '❌ FAIL'}")
    print("=" * 60)
    
    if detection_ok and nav_ok:
        print("\n✅ ALL TESTS PASSED!")
        print("You can now use voice command: 'Hey Rovy, come to me'")
    else:
        print("\n⚠️  Some tests failed. See output above for details.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest aborted by user")
        sys.exit(1)


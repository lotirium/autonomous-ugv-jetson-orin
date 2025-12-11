#!/usr/bin/env python3
"""
Person Detection with OAK-D Camera
Detects people and provides their spatial location (distance, angle) for navigation
"""

import depthai as dai
import numpy as np
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PersonDetection:
    """Information about detected person"""
    confidence: float  # Detection confidence (0-1)
    distance: float  # Distance in meters
    angle_horizontal: float  # Horizontal angle in degrees (-90 to +90, 0 is center)
    angle_vertical: float  # Vertical angle in degrees
    bbox: Tuple[float, float, float, float]  # Normalized bbox (x, y, w, h)
    spatial_coords: Tuple[float, float, float]  # X, Y, Z in meters


class PersonDetector:
    """
    Person detector using OAK-D camera with MobileNet-SSD
    Provides person detection with depth information for navigation
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 camera_hfov: float = 73.0):  # OAK-D horizontal FOV in degrees
        """
        Initialize person detector
        
        Args:
            confidence_threshold: Minimum confidence for person detection
            camera_hfov: Camera horizontal field of view in degrees
        """
        self.confidence_threshold = confidence_threshold
        self.camera_hfov = camera_hfov
        
        self.device = None
        self.detection_queue = None
        self.preview_queue = None
        self.is_running = False
        
        print(f"[PersonDetector] Initialized (confidence >= {confidence_threshold})")
    
    def start(self):
        """Start the person detection pipeline"""
        if self.is_running:
            print("[PersonDetector] Already running")
            return
        
        print("[PersonDetector] Starting OAK-D person detection...")
        
        try:
            # Create pipeline
            pipeline = dai.Pipeline()
            
            # Create color camera
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setPreviewSize(300, 300)  # MobileNet expects 300x300
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setInterleaved(False)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setFps(30)
            
            # Create neural network for object detection
            # Using MobileNet-SSD trained on COCO dataset (class 15 = person)
            detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
            detection_nn.setConfidenceThreshold(self.confidence_threshold)
            detection_nn.setBlobPath("/usr/local/lib/python3.11/dist-packages/depthai/resources/nn/mobilenet-ssd/mobilenet-ssd.blob")
            cam.preview.link(detection_nn.input)
            
            # Create depth
            mono_left = pipeline.create(dai.node.MonoCamera)
            mono_right = pipeline.create(dai.node.MonoCamera)
            stereo = pipeline.create(dai.node.StereoDepth)
            
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(False)
            
            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)
            
            # Link detection to depth for spatial coordinates
            spatial_detection_network = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
            spatial_detection_network.setConfidenceThreshold(self.confidence_threshold)
            spatial_detection_network.setBlobPath("/usr/local/lib/python3.11/dist-packages/depthai/resources/nn/mobilenet-ssd/mobilenet-ssd.blob")
            spatial_detection_network.setBoundingBoxScaleFactor(0.5)
            spatial_detection_network.setDepthLowerThreshold(100)  # 10cm
            spatial_detection_network.setDepthUpperThreshold(5000)  # 5m
            
            cam.preview.link(spatial_detection_network.input)
            stereo.depth.link(spatial_detection_network.inputDepth)
            
            # Create outputs
            xout_detection = pipeline.create(dai.node.XLinkOut)
            xout_detection.setStreamName("detections")
            spatial_detection_network.out.link(xout_detection.input)
            
            xout_preview = pipeline.create(dai.node.XLinkOut)
            xout_preview.setStreamName("preview")
            cam.preview.link(xout_preview.input)
            
            # Start pipeline
            self.device = dai.Device(pipeline)
            self.detection_queue = self.device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            self.preview_queue = self.device.getOutputQueue(name="preview", maxSize=4, blocking=False)
            
            self.is_running = True
            print("[PersonDetector] ✅ OAK-D person detection started!")
            
        except Exception as e:
            print(f"[PersonDetector] ❌ Failed to start: {e}")
            if self.device:
                self.device.close()
                self.device = None
            raise
    
    def stop(self):
        """Stop the person detection pipeline"""
        if not self.is_running:
            return
        
        print("[PersonDetector] Stopping...")
        self.is_running = False
        
        if self.device:
            self.device.close()
            self.device = None
        
        self.detection_queue = None
        self.preview_queue = None
        print("[PersonDetector] Stopped")
    
    def detect_person(self, timeout: float = 0.1) -> Optional[PersonDetection]:
        """
        Detect nearest person in camera view
        
        Args:
            timeout: Timeout for getting detection data
            
        Returns:
            PersonDetection object if person found, None otherwise
        """
        if not self.is_running or not self.detection_queue:
            return None
        
        try:
            # Get latest detections
            detections_data = self.detection_queue.tryGet()
            if detections_data is None:
                return None
            
            # Filter for person class (MobileNet-SSD COCO: class 15 = person)
            PERSON_CLASS_ID = 15
            
            person_detections = []
            for detection in detections_data.detections:
                if detection.label == PERSON_CLASS_ID:
                    # Get spatial coordinates
                    spatial_coords = detection.spatialCoordinates
                    x = spatial_coords.x / 1000.0  # Convert mm to meters
                    y = spatial_coords.y / 1000.0
                    z = spatial_coords.z / 1000.0  # Distance (depth)
                    
                    # Calculate horizontal angle based on x offset and camera FOV
                    # Assuming x is in normalized coordinates (-1 to 1)
                    # Actually, spatial coords are in mm from center
                    bbox_center_x = (detection.xmin + detection.xmax) / 2.0
                    horizontal_angle = (bbox_center_x - 0.5) * self.camera_hfov
                    
                    # Calculate vertical angle similarly
                    bbox_center_y = (detection.ymin + detection.ymax) / 2.0
                    vertical_angle = (0.5 - bbox_center_y) * (self.camera_hfov * 0.75)  # Approximate VFOV
                    
                    person_detections.append(PersonDetection(
                        confidence=detection.confidence,
                        distance=z,
                        angle_horizontal=horizontal_angle,
                        angle_vertical=vertical_angle,
                        bbox=(detection.xmin, detection.ymin, 
                              detection.xmax - detection.xmin, 
                              detection.ymax - detection.ymin),
                        spatial_coords=(x, y, z)
                    ))
            
            if not person_detections:
                return None
            
            # Return nearest person
            nearest = min(person_detections, key=lambda p: p.distance)
            return nearest
            
        except Exception as e:
            print(f"[PersonDetector] Detection error: {e}")
            return None
    
    def scan_for_person(self, 
                        duration: float = 3.0,
                        min_detections: int = 3) -> Optional[PersonDetection]:
        """
        Scan for person with multiple detections for reliability
        
        Args:
            duration: Maximum time to scan in seconds
            min_detections: Minimum number of consistent detections
            
        Returns:
            PersonDetection if person found consistently, None otherwise
        """
        print(f"[PersonDetector] Scanning for person ({duration}s)...")
        
        detections = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            person = self.detect_person()
            if person:
                detections.append(person)
                print(f"[PersonDetector] Person detected at {person.distance:.2f}m, "
                      f"angle {person.angle_horizontal:.1f}° (confidence: {person.confidence:.2f})")
            
            time.sleep(0.1)
        
        if len(detections) >= min_detections:
            # Average the detections for more stable result
            avg_distance = sum(d.distance for d in detections) / len(detections)
            avg_angle = sum(d.angle_horizontal for d in detections) / len(detections)
            best_detection = max(detections, key=lambda d: d.confidence)
            
            result = PersonDetection(
                confidence=sum(d.confidence for d in detections) / len(detections),
                distance=avg_distance,
                angle_horizontal=avg_angle,
                angle_vertical=best_detection.angle_vertical,
                bbox=best_detection.bbox,
                spatial_coords=(
                    sum(d.spatial_coords[0] for d in detections) / len(detections),
                    sum(d.spatial_coords[1] for d in detections) / len(detections),
                    avg_distance
                )
            )
            
            print(f"[PersonDetector] ✅ Person confirmed at {result.distance:.2f}m, "
                  f"angle {result.angle_horizontal:.1f}° ({len(detections)} detections)")
            return result
        else:
            print(f"[PersonDetector] ❌ Person not found (only {len(detections)} detections)")
            return None


def test_person_detector():
    """Test the person detector"""
    print("=" * 60)
    print("PERSON DETECTOR TEST")
    print("=" * 60)
    print("\nStand in front of the camera...")
    print("Press Ctrl+C to stop\n")
    
    detector = PersonDetector(confidence_threshold=0.6)
    
    try:
        detector.start()
        time.sleep(2)  # Let camera stabilize
        
        # Continuous detection
        while True:
            person = detector.detect_person()
            if person:
                print(f"👤 Person: {person.distance:.2f}m @ {person.angle_horizontal:.1f}° "
                      f"(confidence: {person.confidence:.2f})")
            else:
                print("🔍 No person detected")
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        detector.stop()


if __name__ == "__main__":
    test_person_detector()


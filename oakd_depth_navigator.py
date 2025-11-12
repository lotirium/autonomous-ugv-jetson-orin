"""
Oak-D Depth-based Navigator
Uses stereo depth information for precise obstacle detection
"""
import depthai as dai
import cv2
import numpy as np
import random


class OakDDepthCamera:
    """
    Oak-D camera with stereo depth for 3D perception.
    """
    
    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution
        self.device = None
        self.rgb_queue = None
        self.depth_queue = None
        self.pipeline = None
        
    def start(self):
        """Start camera with RGB and depth streams."""
        if self.device is None:
            self.pipeline = dai.Pipeline()
            
            # RGB Camera
            camRgb = self.pipeline.create(dai.node.ColorCamera)
            camRgb.setPreviewSize(self.resolution[0], self.resolution[1])
            camRgb.setInterleaved(False)
            camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            
            # Stereo Depth
            monoLeft = self.pipeline.create(dai.node.MonoCamera)
            monoRight = self.pipeline.create(dai.node.MonoCamera)
            stereo = self.pipeline.create(dai.node.StereoDepth)
            
            monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
            monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            
            # Stereo config
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setExtendedDisparity(False)
            stereo.setSubpixel(False)
            
            # Link cameras to stereo
            monoLeft.out.link(stereo.left)
            monoRight.out.link(stereo.right)
            
            # RGB output
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            camRgb.preview.link(xout_rgb.input)
            
            # Depth output
            xout_depth = self.pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)
            
            # Start device
            self.device = dai.Device(self.pipeline)
            self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            
            print("[Oak-D Depth] Camera started with stereo depth")
    
    def capture_frames(self):
        """
        Capture both RGB and depth frames.
        
        Returns:
            tuple: (rgb_frame, depth_frame) as numpy arrays
        """
        if self.rgb_queue is None or self.depth_queue is None:
            raise RuntimeError("Camera not started")
        
        rgb_msg = self.rgb_queue.get()
        depth_msg = self.depth_queue.get()
        
        rgb_frame = rgb_msg.getCvFrame()
        depth_frame = depth_msg.getFrame()
        
        return rgb_frame, depth_frame
    
    def close(self):
        """Clean up."""
        if self.device:
            self.device.close()
            self.device = None
            self.rgb_queue = None
            self.depth_queue = None
        print("[Oak-D Depth] Camera closed")


class DepthNavigator:
    """
    Navigator using 3D depth information for obstacle avoidance.
    Simple and effective - reacts early to obstacles!
    """
    
    def __init__(self, safe_distance_mm=800):
        """
        Args:
            safe_distance_mm: Minimum safe distance to obstacles in millimeters
        """
        self.safe_distance_mm = safe_distance_mm
        # CRITICAL: Set early warning distance to 1.5x safe distance
        # This gives the rover time to react BEFORE hitting obstacles
        self.warning_distance_mm = int(safe_distance_mm * 1.5)
        self.blocked_distance_mm = int(safe_distance_mm * 0.75)
        print(f"[DepthNav] Initialized (safe: {safe_distance_mm}mm, warning: {self.warning_distance_mm}mm)")
    
    def get_navigation_command(self, rgb_frame, depth_frame):
        """
        Analyze depth map to find safest direction.
        
        Args:
            rgb_frame: RGB image (not used but available)
            depth_frame: Depth map in millimeters
            
        Returns:
            dict: Navigation command
        """
        h, w = depth_frame.shape
        
        # SIMPLIFIED ROBUST APPROACH: Analyze horizontal middle strip only
        # This is where obstacles at robot height appear
        strip_top = int(h * 0.35)  # Middle strip
        strip_bottom = int(h * 0.65)
        depth_strip = depth_frame[strip_top:strip_bottom, :]
        
        # Divide into 5 vertical regions
        regions = {
            'far_left': depth_strip[:, :w//5],
            'left': depth_strip[:, w//5:2*w//5],
            'center': depth_strip[:, 2*w//5:3*w//5],
            'right': depth_strip[:, 3*w//5:4*w//5],
            'far_right': depth_strip[:, 4*w//5:]
        }
        
        # Calculate clearance scores for each region
        scores = {}
        debug_info = {}
        
        for name, region in regions.items():
            # Filter out invalid depth values (0 or very large)
            valid_depths = region[(region > 0) & (region < 5000)]
            
            if len(valid_depths) > 50:  # Need enough valid pixels
                # SIMPLE WORKING APPROACH: Use median distance
                median_dist = np.median(valid_depths)
                
                # SIMPLE LINEAR SCORING (like real robots do):
                # Just normalize distance to 0-1 score
                # Anything > 400mm is usable, prefer farther
                
                MIN_SAFE = 400   # Absolute minimum (real collision risk)
                MAX_CLEAR = 2000  # Fully clear
                
                if median_dist < MIN_SAFE:
                    # Really blocked
                    scores[name] = 0.0
                else:
                    # Linear score: 400mm=0.3, 800mm=0.6, 1200mm=0.75, 2000mm+=1.0
                    normalized = (median_dist - MIN_SAFE) / (MAX_CLEAR - MIN_SAFE)
                    scores[name] = 0.3 + min(normalized, 1.0) * 0.7
                
                debug_info[name] = {'median': int(median_dist), 'count': len(valid_depths)}
            else:
                scores[name] = 0.0
                debug_info[name] = {'avg': 0, 'min': 0, 'count': len(valid_depths)}
        
        # --- SAFETY-FIRST DECISION LOGIC (VFH-inspired) ---
        # Step 1: SAFETY FILTER - only consider actions above minimum safety threshold
        # Step 2: Among safe actions, prefer forward with moderate bias
        
        center_score = scores.get('center', 0.0)
        left_score = max(scores.get('left', 0), scores.get('far_left', 0))
        right_score = max(scores.get('right', 0), scores.get('far_right', 0))
        
        # SAFETY THRESHOLD: Minimum clearance required to consider an action safe
        SAFETY_THRESHOLD = 0.35  # 35% minimum - allows navigation in tight spaces
        
        # Filter safe actions
        safe_actions = {}
        if center_score >= SAFETY_THRESHOLD:
            safe_actions['forward'] = center_score
        if left_score >= SAFETY_THRESHOLD:
            safe_actions['left'] = left_score
        if right_score >= SAFETY_THRESHOLD:
            safe_actions['right'] = right_score
        
        # Make decision
        if not safe_actions:
            # NO SAFE PATHS - must turn to find exit
            action = random.choice(['left', 'right'])
            reasoning = f'No safe path (C={int(center_score*100)}% L={int(left_score*100)}% R={int(right_score*100)}%) - exploring'
            
        elif 'forward' in safe_actions:
            # Forward is safe - prefer it with moderate bias
            forward_score = safe_actions['forward']
            best_side = max(safe_actions.get('left', 0), safe_actions.get('right', 0))
            
            # Go forward if it's competitive (within 85% of best side)
            if forward_score >= best_side * 0.85:
                action = 'forward'
                reasoning = f'Forward safe & competitive (C={int(forward_score*100)}% ≥ 85% of best={int(best_side*100)}%)'
            else:
                # Side is significantly better - take it
                if safe_actions.get('left', 0) > safe_actions.get('right', 0):
                    action = 'left'
                    reasoning = f'Left much clearer (L={int(left_score*100)}% >> C={int(forward_score*100)}%)'
                else:
                    action = 'right'
                    reasoning = f'Right much clearer (R={int(right_score*100)}% >> C={int(forward_score*100)}%)'
        else:
            # Forward not safe - choose best side
            if safe_actions.get('left', 0) > safe_actions.get('right', 0):
                action = 'left'
                reasoning = f'Center blocked (C={int(center_score*100)}%), evading left (L={int(left_score*100)}%)'
            else:
                action = 'right'
                reasoning = f'Center blocked (C={int(center_score*100)}%), evading right (R={int(right_score*100)}%)'
        
        # Determine speed based on clearance - BE VERY CAUTIOUS
        current_path_score = center_score if action == 'forward' else max(left_score, right_score)
        
        if current_path_score > 0.8:
            speed = 'slow'  # Always go slow for safety
            distance = 0.3
        elif current_path_score > 0.6:
            speed = 'slow'
            distance = 0.2
        else:
            speed = 'slow'
            distance = 0.15
        
        # Debug: show decision reasoning occasionally
        if random.random() < 0.15:  # 15% of the time
            print(f"[DepthNav] Scores: C={int(center_score*100)}% L={int(scores.get('left',0)*100)}% R={int(scores.get('right',0)*100)}% -> {action.upper()}")
        
        return {
            'action': action,
            'speed': speed,
            'distance': distance,
            'reasoning': reasoning,
            'scores': scores  # Return 5-zone scores for LLaVA
        }
    
    # !!! rotate_and_scan FUNCTION REMOVED !!!
    # It conflicted with _capture_thread
    
    def cleanup(self):
        print("[DepthNav] Closed")

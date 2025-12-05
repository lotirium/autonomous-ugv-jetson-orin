import { DEFAULT_CLOUD_URL } from './cloud-api';

export type DetectedGesture = 'like' | 'heart' | 'none';

interface GestureDetectionResponse {
  gesture: DetectedGesture;
  confidence: number;
  status: string;
}

/**
 * Service for detecting hand gestures from images
 * Uses the cloud server's MediaPipe-based gesture detection
 */
export class GestureDetectionService {
  private baseUrl: string;
  private detectionInterval: number = 500; // ms between detections
  private lastDetectionTime: number = 0;

  constructor(baseUrl: string = DEFAULT_CLOUD_URL) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  /**
   * Detect gesture from image data URI or base64 string
   */
  async detectGesture(imageDataUri: string): Promise<DetectedGesture> {
    try {
      // Throttle requests to avoid overwhelming the server
      const now = Date.now();
      if (now - this.lastDetectionTime < this.detectionInterval) {
        return 'none';
      }
      this.lastDetectionTime = now;

      // Convert data URI to blob
      const base64Data = imageDataUri.split(',')[1] || imageDataUri;
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'image/jpeg' });

      // Create form data
      const formData = new FormData();
      formData.append('file', blob, 'gesture.jpg');

      // Send to server
      const response = await fetch(`${this.baseUrl}/gesture/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        console.warn('Gesture detection failed:', response.statusText);
        return 'none';
      }

      const result: GestureDetectionResponse = await response.json();
      
      // Only return gesture if confidence is above threshold
      if (result.confidence > 0.5) {
        return result.gesture;
      }

      return 'none';
    } catch (error) {
      console.error('Gesture detection error:', error);
      return 'none';
    }
  }

  /**
   * Update base URL for gesture detection service
   */
  updateBaseUrl(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }
}

// Singleton instance
let gestureDetectionService: GestureDetectionService | null = null;

export function getGestureDetectionService(baseUrl?: string): GestureDetectionService {
  if (!gestureDetectionService) {
    gestureDetectionService = new GestureDetectionService(baseUrl);
  } else if (baseUrl) {
    gestureDetectionService.updateBaseUrl(baseUrl);
  }
  return gestureDetectionService;
}

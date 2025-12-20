"""
FloodWatch Calibration Tool
============================
HSV Color Calibration Interface for generating calibration.json.

This script runs on a PC to help calibrate the HSV color ranges for
detecting Yellow, Orange, and Red markers in the specific environment.

Controls:
---------
- 'y': Select Yellow for calibration
- 'o': Select Orange for calibration  
- 'r': Select Red for calibration
- 's': Save current calibration to file
- 'q': Quit and save calibration
- ESC: Quit without saving

Usage:
------
1. Run: python calibrate.py
2. Point camera at the colored markers
3. Press 'y', 'o', or 'r' to select a color
4. Adjust sliders until only that color appears white in the mask
5. Press 's' to save the current color's range
6. Repeat for all colors
7. Press 'q' to quit and save final calibration

Alternatively, use an image file:
    python calibrate.py --image path/to/markers.jpg
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Optional


class HSVCalibrator:
    """HSV Color Calibration Tool for FloodWatch markers."""
    
    WINDOW_NAME = "FloodWatch Calibration"
    MASK_WINDOW = "Color Mask"
    
    # Default HSV ranges (starting points)
    DEFAULT_RANGES = {
        "yellow": {"lower": [25, 120, 100], "upper": [35, 255, 255]},
        "orange": {"lower": [5, 120, 100], "upper": [25, 255, 255]},
        "red": {"lower": [170, 120, 100], "upper": [180, 255, 255]}
    }
    
    def __init__(self, output_file: str = "calibration.json"):
        self.output_file = Path(output_file)
        self.calibration = self.DEFAULT_RANGES.copy()
        self.current_color = "yellow"
        
        # Current slider values
        self.low_h = 0
        self.low_s = 0
        self.low_v = 0
        self.high_h = 179
        self.high_s = 255
        self.high_v = 255
        
        self._setup_windows()
    
    def _setup_windows(self):
        """Create windows and trackbars."""
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.namedWindow(self.MASK_WINDOW)
        
        # Create trackbars for HSV adjustment
        cv2.createTrackbar("Low H", self.WINDOW_NAME, 0, 179, self._on_trackbar)
        cv2.createTrackbar("High H", self.WINDOW_NAME, 179, 179, self._on_trackbar)
        cv2.createTrackbar("Low S", self.WINDOW_NAME, 0, 255, self._on_trackbar)
        cv2.createTrackbar("High S", self.WINDOW_NAME, 255, 255, self._on_trackbar)
        cv2.createTrackbar("Low V", self.WINDOW_NAME, 0, 255, self._on_trackbar)
        cv2.createTrackbar("High V", self.WINDOW_NAME, 255, 255, self._on_trackbar)
    
    def _on_trackbar(self, val):
        """Trackbar callback (required by OpenCV but values read directly)."""
        pass
    
    def _read_trackbars(self):
        """Read current trackbar values."""
        self.low_h = cv2.getTrackbarPos("Low H", self.WINDOW_NAME)
        self.high_h = cv2.getTrackbarPos("High H", self.WINDOW_NAME)
        self.low_s = cv2.getTrackbarPos("Low S", self.WINDOW_NAME)
        self.high_s = cv2.getTrackbarPos("High S", self.WINDOW_NAME)
        self.low_v = cv2.getTrackbarPos("Low V", self.WINDOW_NAME)
        self.high_v = cv2.getTrackbarPos("High V", self.WINDOW_NAME)
    
    def _set_trackbars(self, lower: list, upper: list):
        """Set trackbar values from a color range."""
        cv2.setTrackbarPos("Low H", self.WINDOW_NAME, lower[0])
        cv2.setTrackbarPos("High H", self.WINDOW_NAME, upper[0])
        cv2.setTrackbarPos("Low S", self.WINDOW_NAME, lower[1])
        cv2.setTrackbarPos("High S", self.WINDOW_NAME, upper[1])
        cv2.setTrackbarPos("Low V", self.WINDOW_NAME, lower[2])
        cv2.setTrackbarPos("High V", self.WINDOW_NAME, upper[2])
    
    def select_color(self, color: str):
        """Switch to calibrating a different color."""
        if color in self.calibration:
            self.current_color = color
            bounds = self.calibration[color]
            self._set_trackbars(bounds["lower"], bounds["upper"])
            print(f"\n>>> Now calibrating: {color.upper()}")
            print(f"    Adjust sliders until only {color} markers are white in mask")
    
    def save_current_color(self):
        """Save current trackbar values to the selected color."""
        self._read_trackbars()
        self.calibration[self.current_color] = {
            "lower": [self.low_h, self.low_s, self.low_v],
            "upper": [self.high_h, self.high_s, self.high_v]
        }
        print(f"    Saved {self.current_color}: L={[self.low_h, self.low_s, self.low_v]}, "
              f"U={[self.high_h, self.high_s, self.high_v]}")
    
    def save_to_file(self):
        """Write calibration to JSON file."""
        with open(self.output_file, "w") as f:
            json.dump(self.calibration, f, indent=2)
        print(f"\n>>> Calibration saved to: {self.output_file.absolute()}")
    
    def load_from_file(self):
        """Load existing calibration if available."""
        if self.output_file.exists():
            with open(self.output_file, "r") as f:
                self.calibration = json.load(f)
            print(f">>> Loaded existing calibration from: {self.output_file}")
            return True
        return False
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Process a frame and return the annotated frame and mask.
        
        Returns:
            tuple: (annotated_frame, binary_mask)
        """
        self._read_trackbars()
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask with current slider values
        lower = np.array([self.low_h, self.low_s, self.low_v])
        upper = np.array([self.high_h, self.high_s, self.high_v])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Count pixels for feedback
        pixel_count = cv2.countNonZero(mask)
        
        # Draw info overlay
        info_frame = frame.copy()
        cv2.putText(info_frame, f"Color: {self.current_color.upper()}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(info_frame, f"Pixels: {pixel_count}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(info_frame, f"L: [{self.low_h},{self.low_s},{self.low_v}]", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_frame, f"U: [{self.high_h},{self.high_s},{self.high_v}]", 
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(info_frame, "Y/O/R: Select color | S: Save | Q: Quit", 
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return info_frame, mask
    
    def run_with_camera(self, camera_index: int = 0):
        """Run calibration with live camera feed."""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("ERROR: Could not open camera")
            return False
        
        print("\n" + "=" * 50)
        print("FloodWatch HSV Calibration Tool")
        print("=" * 50)
        print("\nControls:")
        print("  Y - Calibrate Yellow marker")
        print("  O - Calibrate Orange marker")
        print("  R - Calibrate Red marker")
        print("  S - Save current color's range")
        print("  Q - Quit and save to file")
        print("  ESC - Quit without saving")
        print("=" * 50)
        
        # Load existing calibration if available
        self.load_from_file()
        self.select_color("yellow")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Failed to read frame")
                    break
                
                display_frame, mask = self.process_frame(frame)
                
                cv2.imshow(self.WINDOW_NAME, display_frame)
                cv2.imshow(self.MASK_WINDOW, mask)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('y'):
                    self.select_color("yellow")
                elif key == ord('o'):
                    self.select_color("orange")
                elif key == ord('r'):
                    self.select_color("red")
                elif key == ord('s'):
                    self.save_current_color()
                elif key == ord('q'):
                    self.save_current_color()
                    self.save_to_file()
                    break
                elif key == 27:  # ESC
                    print("\n>>> Exiting without saving...")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return True
    
    def run_with_image(self, image_path: str):
        """Run calibration with a static image."""
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"ERROR: Could not load image: {image_path}")
            return False
        
        print("\n" + "=" * 50)
        print("FloodWatch HSV Calibration Tool (Image Mode)")
        print("=" * 50)
        print(f"\nLoaded image: {image_path}")
        print("\nControls:")
        print("  Y - Calibrate Yellow marker")
        print("  O - Calibrate Orange marker")
        print("  R - Calibrate Red marker")
        print("  S - Save current color's range")
        print("  Q - Quit and save to file")
        print("  ESC - Quit without saving")
        print("=" * 50)
        
        self.load_from_file()
        self.select_color("yellow")
        
        try:
            while True:
                display_frame, mask = self.process_frame(frame)
                
                cv2.imshow(self.WINDOW_NAME, display_frame)
                cv2.imshow(self.MASK_WINDOW, mask)
                
                key = cv2.waitKey(50) & 0xFF
                
                if key == ord('y'):
                    self.select_color("yellow")
                elif key == ord('o'):
                    self.select_color("orange")
                elif key == ord('r'):
                    self.select_color("red")
                elif key == ord('s'):
                    self.save_current_color()
                elif key == ord('q'):
                    self.save_current_color()
                    self.save_to_file()
                    break
                elif key == 27:  # ESC
                    print("\n>>> Exiting without saving...")
                    break
        finally:
            cv2.destroyAllWindows()
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="FloodWatch HSV Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calibrate.py                    # Use default camera
  python calibrate.py --camera 1         # Use camera index 1
  python calibrate.py --image test.jpg   # Use static image
  python calibrate.py --output my_cal.json  # Custom output file
        """
    )
    parser.add_argument(
        "--camera", "-c", 
        type=int, 
        default=0, 
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--image", "-i", 
        type=str, 
        default=None, 
        help="Path to image file (optional, uses camera if not specified)"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="calibration.json", 
        help="Output calibration file (default: calibration.json)"
    )
    
    args = parser.parse_args()
    
    calibrator = HSVCalibrator(output_file=args.output)
    
    if args.image:
        calibrator.run_with_image(args.image)
    else:
        calibrator.run_with_camera(args.camera)


if __name__ == "__main__":
    main()

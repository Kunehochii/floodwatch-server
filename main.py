"""
FloodWatch Backend Server
=========================
Real-time water level monitoring using Computer Vision (OpenCV).
Detects the "disappearance" of colored markers (Yellow, Orange, Red) to trigger flood alerts.

Run with: python main.py
Or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Generator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FloodWatch API",
    description="Real-time water level monitoring system using Computer Vision",
    version="1.1.0"
)

# Enable CORS for Frontend Access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CONFIG_FILE = Path(__file__).parent / "calibration.json"
PIXEL_THRESHOLD = 300  # Minimum pixels to consider a color "present"

# Global State
current_status = {"level": 0, "message": "Initializing", "presence": {}}
calibration_config = {}


def load_calibration() -> dict:
    """
    Load HSV color ranges from calibration.json.
    This file is REQUIRED for the detection logic to function.
    """
    global calibration_config
    
    if not CONFIG_FILE.exists():
        logger.warning(f"calibration.json not found at {CONFIG_FILE}. Using defaults.")
        # Default calibration values - should be replaced with actual calibration
        calibration_config = {
            "yellow": {"lower": [25, 120, 100], "upper": [35, 255, 255]},
            "orange": {"lower": [5, 120, 100], "upper": [25, 255, 255]},
            "red": {"lower": [170, 120, 100], "upper": [180, 255, 255]}
        }
    else:
        with open(CONFIG_FILE, "r") as f:
            calibration_config = json.load(f)
        logger.info(f"Loaded calibration from {CONFIG_FILE}")
    
    return calibration_config


def detect_water_level(frame: np.ndarray) -> tuple[int, str, dict]:
    """
    Detect water level based on marker visibility.
    
    The "Disappearance" Algorithm:
    - Step 1: Convert Frame to HSV
    - Step 2: Apply Masks for Yellow, Orange, Red using calibration ranges
    - Step 3: Count non-zero pixels for each mask
    - Step 4: Compare count against threshold (color "missing" if below)
    - Step 5: Determine Flood Level based on which colors are missing
    
    Returns:
        tuple: (level, message, presence_dict)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    presence = {}
    
    # Iterate through colors in calibration config
    for color, bounds in calibration_config.items():
        lower = np.array(bounds["lower"])
        upper = np.array(bounds["upper"])
        mask = cv2.inRange(hsv, lower, upper)
        count = cv2.countNonZero(mask)
        presence[color] = count > PIXEL_THRESHOLD

    # Logic: Disappearance = Trigger
    # Yellow is bottom (Level 1), Orange is middle (Level 2), Red is top (Level 3)
    yellow_visible = presence.get("yellow", False)
    orange_visible = presence.get("orange", False)
    red_visible = presence.get("red", False)
    
    if not yellow_visible and not orange_visible and not red_visible:
        return 3, "FLOOD: Evacuate Immediately", presence
    elif not yellow_visible and not orange_visible:
        return 2, "CRITICAL: Water at Orange Level", presence
    elif not yellow_visible:
        return 1, "WARNING: Water at Yellow Level", presence
    else:
        return 0, "NORMAL: Monitoring", presence


def generate_frames() -> Generator[bytes, None, None]:
    """
    Generator function for MJPEG video stream.
    Captures frames, runs detection, and yields encoded JPEG bytes.
    """
    global current_status
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    logger.info("Camera opened successfully, starting frame capture")
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to read frame")
                break
            
            # Run detection
            level, msg, presence = detect_water_level(frame)
            current_status = {
                "level": level,
                "message": msg,
                "presence": presence
            }
            
            # Draw status overlay on frame (optional visual feedback)
            cv2.putText(
                frame, 
                f"Level: {level} - {msg}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Encode for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        logger.info("Camera released")


# =============================================================================
# API Endpoints
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load calibration on server startup."""
    load_calibration()


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "FloodWatch API", "version": "1.1.0"}


@app.get("/video_feed", tags=["Stream"])
def video_feed():
    """
    MJPEG video stream endpoint.
    Returns a multipart/x-mixed-replace response for live video streaming.
    
    Usage: Set as `src` attribute of an <img> tag.
    """
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status", tags=["Status"])
def get_status():
    """
    Get current flood detection status.
    
    Returns:
        JSON object with:
        - level (int): 0=Normal, 1=Warning, 2=Critical, 3=Flood
        - message (str): Human-readable status message
        - presence (dict): Visibility status of each color marker
    """
    return JSONResponse(content=current_status)


@app.get("/calibration", tags=["Configuration"])
def get_calibration():
    """
    Get current HSV calibration values.
    
    Returns:
        JSON object with HSV ranges for yellow, orange, and red markers.
    """
    return JSONResponse(content=calibration_config)


@app.post("/calibration/reload", tags=["Configuration"])
def reload_calibration():
    """
    Reload calibration from calibration.json file.
    Use this after updating the calibration file without restarting the server.
    """
    config = load_calibration()
    return JSONResponse(content={
        "status": "reloaded",
        "config": config
    })


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    # Run on 0.0.0.0 to be accessible on local network
    uvicorn.run(app, host="0.0.0.0", port=8000)

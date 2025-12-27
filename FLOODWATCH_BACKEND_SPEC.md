# FloodWatch Backend API Specification

**Version:** 1.1.0  
**Base URL:** `http://<PI_IP_ADDRESS>:8000`  
**Protocol:** HTTP/1.1

---

## Overview

The FloodWatch backend provides real-time water level monitoring through computer vision. It exposes a REST API for status retrieval and an MJPEG stream for live video feed.

---

## Authentication

Currently, no authentication is required. CORS is enabled for all origins.

> **Production Note:** Restrict `allow_origins` in the CORS middleware to your frontend's domain.

---

## Endpoints

### Health Check

```
GET /
```

**Description:** Verify server is running.

**Response:**
```json
{
  "status": "ok",
  "service": "FloodWatch API",
  "version": "1.1.0"
}
```

---

### Video Feed (MJPEG Stream)

```
GET /video_feed
```

**Description:** Live video stream from the camera with detection overlay.

**Response:**
- **Content-Type:** `multipart/x-mixed-replace; boundary=frame`
- **Body:** Continuous MJPEG frames

**Frontend Usage:**
```html
<img src="http://<API_URL>/video_feed" alt="Live Stream" />
```

**Notes:**
- Each frame includes a text overlay showing current detection level
- Stream continues indefinitely while server is running
- Browser handles frame updates automatically via `multipart/x-mixed-replace`

---

### Detection Status

```
GET /status
```

**Description:** Get current flood detection status.

**Response:**
```json
{
  "level": 0,
  "message": "NORMAL: Monitoring",
  "presence": {
    "yellow": true,
    "orange": true,
    "red": true
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `level` | integer | Flood level (0-3) |
| `message` | string | Human-readable status message |
| `presence` | object | Visibility status of each color marker |

**Flood Levels:**

| Level | Name | Condition | Message |
|-------|------|-----------|---------|
| 0 | Normal | All markers visible | "NORMAL: Monitoring" |
| 1 | Warning | Yellow missing | "WARNING: Water at Yellow Level" |
| 2 | Critical | Yellow + Orange missing | "CRITICAL: Water at Orange Level" |
| 3 | Flood | All markers missing | "FLOOD: Evacuate Immediately" |

**Frontend Usage:**
```javascript
// Poll every 1 second
setInterval(async () => {
  const response = await fetch(`${API_URL}/status`);
  const data = await response.json();
  // data.level, data.message, data.presence
}, 1000);
```

---

### Get Calibration

```
GET /calibration
```

**Description:** Retrieve current HSV calibration values.

**Response:**
```json
{
  "yellow": {
    "lower": [25, 120, 100],
    "upper": [35, 255, 255]
  },
  "orange": {
    "lower": [5, 120, 100],
    "upper": [25, 255, 255]
  },
  "red": {
    "lower": [170, 120, 100],
    "upper": [180, 255, 255]
  }
}
```

**HSV Range Format:**
- `lower`: [Hue, Saturation, Value] minimum bounds
- `upper`: [Hue, Saturation, Value] maximum bounds
- Hue: 0-179, Saturation: 0-255, Value: 0-255

---

### Reload Calibration

```
POST /calibration/reload
```

**Description:** Reload calibration from `calibration.json` file without restarting the server.

**Response:**
```json
{
  "status": "reloaded",
  "config": {
    "yellow": { "lower": [...], "upper": [...] },
    "orange": { "lower": [...], "upper": [...] },
    "red": { "lower": [...], "upper": [...] }
  }
}
```

**Use Case:** After updating `calibration.json` manually or via the calibration tool, call this endpoint to apply new values.

---

## Error Handling

### Common Errors

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 500 | Internal server error (e.g., camera failure) |

**Error Response Format:**
```json
{
  "detail": "Error description"
}
```

---

## Frontend Integration Guide

### Environment Configuration

Create a `.env` file in your frontend project:

```env
VITE_API_URL=http://192.168.1.X:8000
```

### React/Vite Implementation

```jsx
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Video Feed Component
function VideoFeed() {
  return (
    <img 
      src={`${API_URL}/video_feed`} 
      alt="Live Stream"
      className="w-full h-full object-cover"
    />
  );
}

// Status Polling Hook
function useFloodStatus() {
  const [status, setStatus] = useState({ level: 0, message: "Connecting..." });

  useEffect(() => {
    const interval = setInterval(() => {
      fetch(`${API_URL}/status`)
        .then(res => res.json())
        .then(data => setStatus(data))
        .catch(err => console.error("Connection Error:", err));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return status;
}
```

### Status Level Styling

```javascript
const levelColors = {
  0: "bg-green-500",   // Normal
  1: "bg-yellow-500",  // Warning
  2: "bg-orange-500",  // Critical
  3: "bg-red-600"      // Flood
};

const levelLabels = {
  0: "NORMAL",
  1: "WARNING",
  2: "CRITICAL",
  3: "FLOOD ALERT"
};
```

---

## Deployment Notes

### Server Startup

```bash
# Development (with auto-reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production
python main.py
```

### Prerequisites

1. `calibration.json` must exist in the same directory as `main.py`
2. Camera must be accessible at index 0 (or configure accordingly)
3. Required packages: `fastapi`, `uvicorn`, `opencv-python`, `numpy`

### Network Configuration

- Server binds to `0.0.0.0:8000` (accessible from LAN)
- Ensure port 8000 is open in firewall
- Frontend and backend must be on the same network

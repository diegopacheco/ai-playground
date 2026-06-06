import asyncio
import functools
import http.server
import json
import os
import socketserver
import threading

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
MODEL = os.path.join(HERE, "pose_landmarker.task")
HTTP_PORT = 8000
WS_PORT = 8765

FACE_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def new_detector():
    base = mp_python.BaseOptions(model_asset_path=MODEL)
    options = vision.PoseLandmarkerOptions(
        base_options=base,
        num_poses=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def face_box(lm):
    xs, ys = [], []
    for i in FACE_IDS:
        p = lm[i]
        if getattr(p, "visibility", 1.0) >= 0.3:
            xs.append(p.x)
            ys.append(p.y)
    if not xs:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad_x = (x1 - x0) * 0.55 + 0.02
    pad_y = (y1 - y0) * 0.6 + 0.02
    fx = max(0.0, x0 - pad_x)
    fy = max(0.0, y0 - pad_y * 1.6)
    fw = min(1.0, x1 + pad_x) - fx
    fh = min(1.0, y1 + pad_y * 0.7) - fy
    return {"x": fx, "y": fy, "w": fw, "h": fh}


def read_pose(detector, data):
    buf = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return {"present": False}
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(image)
    if not result.pose_landmarks:
        return {"present": False}
    lm = result.pose_landmarks[0]
    cx = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2.0
    cy = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2.0
    out = {
        "present": True,
        "x": min(1.0, max(0.0, 1.0 - cx)),
        "y": min(1.0, max(0.0, cy)),
    }
    fb = face_box(lm)
    if fb:
        out["face"] = fb
    return out


async def handle(ws):
    detector = new_detector()
    loop = asyncio.get_event_loop()
    try:
        async for message in ws:
            if not isinstance(message, (bytes, bytearray)):
                continue
            payload = await loop.run_in_executor(None, read_pose, detector, bytes(message))
            await ws.send(json.dumps(payload))
    except websockets.ConnectionClosed:
        pass
    finally:
        detector.close()


def serve_http():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB_DIR)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", HTTP_PORT), handler) as httpd:
        httpd.serve_forever()


async def main():
    threading.Thread(target=serve_http, daemon=True).start()
    async with websockets.serve(handle, "", WS_PORT, max_size=2 ** 22):
        print(f"game at http://localhost:{HTTP_PORT}")
        print(f"pose-tracking websocket on ws://localhost:{WS_PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

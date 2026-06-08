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
MODEL = os.path.join(HERE, "hand_landmarker.task")
HTTP_PORT = 8000
WS_PORT = 8765


def new_detector():
    base = mp_python.BaseOptions(model_asset_path=MODEL)
    options = vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=2,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)


def classify(lm):
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y
    up = index_up + middle_up + ring_up + pinky_up
    fist = up == 0
    open_palm = up >= 3
    cx = (lm[0].x + lm[5].x + lm[17].x) / 3.0
    cy = (lm[0].y + lm[5].y + lm[17].y) / 3.0
    x = min(1.0, max(0.0, 1.0 - cx))
    y = min(1.0, max(0.0, cy))
    scale = ((lm[5].x - lm[17].x) ** 2 + (lm[5].y - lm[17].y) ** 2) ** 0.5
    side = "L" if x < 0.5 else "R"
    return {"x": x, "y": y, "side": side, "fist": fist, "open": open_palm, "scale": scale}


def hands_from_frame(detector, data):
    buf = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return {"hands": []}
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(image)
    if not result.hand_landmarks:
        return {"hands": []}
    return {"hands": [classify(lm) for lm in result.hand_landmarks]}


async def handle(ws):
    detector = new_detector()
    loop = asyncio.get_event_loop()
    try:
        async for message in ws:
            if not isinstance(message, (bytes, bytearray)):
                continue
            payload = await loop.run_in_executor(None, hands_from_frame, detector, bytes(message))
            await ws.send(json.dumps(payload))
    except websockets.ConnectionClosed:
        pass
    finally:
        detector.close()


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()


def serve_http():
    handler = functools.partial(NoCacheHandler, directory=WEB_DIR)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", HTTP_PORT), handler) as httpd:
        httpd.serve_forever()


async def main():
    threading.Thread(target=serve_http, daemon=True).start()
    async with websockets.serve(handle, "", WS_PORT, max_size=2 ** 22):
        print(f"air lango at http://localhost:{HTTP_PORT}")
        print(f"hand-tracking websocket on ws://localhost:{WS_PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

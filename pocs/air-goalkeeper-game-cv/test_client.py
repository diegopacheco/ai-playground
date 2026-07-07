import asyncio
import json
import sys

import cv2
import numpy as np
import websockets


async def main():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        print("encode failed")
        sys.exit(1)
    async with websockets.connect("ws://localhost:8765", max_size=2 ** 22) as ws:
        await ws.send(buf.tobytes())
        resp = await asyncio.wait_for(ws.recv(), timeout=15)
        obj = json.loads(resp)
        if "hands" not in obj:
            print("bad response", obj)
            sys.exit(1)
        print("websocket pipeline ok ->", obj)


asyncio.run(main())

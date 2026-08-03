# Fly Catcher

Fly Catcher is a local 8-bit kitchen arcade game controlled by an iPhone accelerometer. The Mac runs the game in a browser on loopback only. The native iPhone controller sends motion and snap events directly to UDP port `5005` over your private LAN.

No third-party packages, cloud services, accounts, or public web endpoints are used.

## Requirements

- macOS with Node.js 20 or newer
- Xcode 16 or newer
- iPhone running iOS 17 or newer
- Mac and iPhone connected to the same private Wi-Fi network

## Start the game

```bash
chmod +x start.sh stop.sh test.sh
./start.sh
```

Open the address printed by `start.sh` on the Mac. The preferred address is `http://127.0.0.1:8080`. If that port is occupied, the server selects the next free port and prints it.

Find the Mac private IPv4 address:

```bash
ipconfig getifaddr en0
```

If Wi-Fi is not on `en0`, inspect the active interfaces:

```bash
ifconfig
```

## Install the iPhone controller

1. Open `ios/FlyCatcherController.xcodeproj` in Xcode.
2. Select the `FlyCatcherController` target.
3. In Signing & Capabilities, select your Apple development team.
4. Connect the iPhone, select it as the run destination, and press Run.
5. Approve the local-network permission on the iPhone.
6. Open the game page on the Mac.
7. Scan its QR code with the iPhone Camera.
8. Open the private pairing page shown by Camera.
9. Tap `Open UDP Controller`.

The QR code contains a standard HTTP address hosted only on the Mac private network. The page opens the installed controller with the detected Mac private IPv4 address and selected UDP port. Manual address and port entry remain available in the controller.

The game unlocks its Start button only after receiving an iPhone UDP packet. The iPhone app displays live X, Y, and Z acceleration values. Hold it in a comfortable portrait position, wait for the green linked state, start a round on the Mac, then press `Center aim`. Tilt the phone to move the swatter. Push the phone forward quickly to snap, or press the large `Snap` button.

The Mac firewall may ask whether Node can receive incoming connections. Allow local connections so the iPhone UDP packets can reach the game.

## Controls and scoring

- Tilt the iPhone to aim.
- Make a quick forward motion to strike.
- Press `Snap` on the iPhone for a manual strike.
- Arrow keys and Space provide Mac keyboard controls.
- A hit awards 100 points plus a growing chain bonus.
- A miss costs 25 points and resets the chain.
- Each round lasts 60 seconds.

## Stop the game

```bash
./stop.sh
```

## Test

```bash
./test.sh
```

The test checks the game page, status endpoint, browser event stream, loopback HTTP address, private UDP motion packets, occupied-port fallback, saved-state cleanup, QR structure, fixed viewport, and controller gate.

## Network boundary

The HTTP game server binds only to `127.0.0.1`, so another machine cannot open it. UDP binds to the selected port on the Mac because the iPhone must reach it, but packets are accepted only from loopback, link-local, and private IPv4 ranges. The server rejects public source addresses, malformed JSON, unknown packet types, and packets larger than 1024 bytes.

The preferred ports are HTTP `8080` and UDP `5005`. When either is occupied, the server checks subsequent ports until it finds a free one. The selected ports and process ID are stored in `.fly-catcher.state`, allowing `start.sh`, `stop.sh`, and `test.sh` to use the same values.

Supported UDP payloads:

```json
{"type":"motion","ax":0.12,"ay":-0.44,"az":0.91}
```

```json
{"type":"snap"}
```

Change ports for one run with matching values on the Mac and iPhone:

```bash
GAME_PORT=8090 UDP_PORT=5006 ./start.sh
```

If the Mac has several private network interfaces, select the pairing address explicitly:

```bash
CONTROLLER_HOST=192.168.1.20 ./start.sh
```

# Seagotchi

A 16-bit style virtual pet built with Three.js and WebGL. Care for Blubber, a California sea lion resting on a rocky Pacific cove while waves roll and nearby sea lions swim past.

## Features

- Animated low-poly WebGL ocean and coastal scene
- Food, sleep, and happiness meters
- Fish feeding, sleeping, and petting actions
- Progressive growth after every fish
- Chonkers achievement for dedicated feeding
- Mouse and touch scene rotation
- Responsive handheld-console interface

## Requirements

- Node.js 20.19 or newer
- npm

## Run

```bash
./start.sh
```

Seagotchi starts at [http://localhost:4242](http://localhost:4242). If that port is occupied, the script selects the next available port and prints the address.

## Stop

```bash
./stop.sh
```

The stop script terminates the Seagotchi process running on the selected port.

## Production build

```bash
npm install
npm run build
```

The optimized files are written to `dist`.

# Dustline

Dustline is a fast browser first-person shooter set across the occupied streets, factories, warehouses, and towers of Port Meridian. Fight five armed bots, collect stronger weapons, and reach 12 eliminations before the five-minute limit.

## Run

```bash
./start.sh
```

Open the URL printed by the script. It starts at port `8080` and automatically selects the next free port when needed. The selected port is remembered for shutdown and testing.

```bash
./stop.sh
```

Set a different starting port when needed:

```bash
DUSTLINE_PORT=8090 ./start.sh
./stop.sh
```

## Controls

Desktop:

- `W`, `A`, `S`, `D`: move
- `Shift`: sprint
- Mouse: aim
- Left mouse button: fire
- `R`: reload
- `E`: collect a nearby weapon
- `1`, `2`, `3`: switch owned weapons
- `Esc`: release the pointer and pause

iPhone and iPad Safari:

- Use landscape orientation
- Left stick: move
- Drag the right side: aim
- Round right button: fire
- `RUN`: sprint
- `USE`: collect a nearby weapon
- `R`: reload

Open the game using the Mac's local network address to play from an iPhone or iPad on the same Wi-Fi network. The terminal prints the desktop URL; replace `localhost` with the Mac's IP address.

## Goal

- Win by reaching 12 eliminations
- Lose when health reaches zero or the timer expires
- Bots pursue, strafe, take cover around solid geometry, fire at visible players, and return after a short delay
- Enemies use distinct animated character models with movement, weapon-holding, firing, and death states
- The map includes city blocks, roads, towers, warehouses, streetlights, construction barriers, factory machinery, pipes, catwalks, a crane, and grounded supply props
- First-person arms hold every equipped weapon and move with recoil and locomotion
- The VX-9 sidearm is available at deployment
- The AR-4 carbine and M12 breacher are placed in the arena

## Performance

The renderer targets 60 FPS. Mobile devices use a capped pixel ratio, reduced shadow work, low-poly enemies, shared materials, short-lived combat effects, and local compressed models. Safari performance improves when other browser tabs are closed and Low Power Mode is disabled.

## Assets

- [Kenney Blaster Kit](https://kenney.nl/assets/blaster-kit): weapon and crate models, CC0
- [Kenney City Kit Industrial](https://kenney.nl/assets/city-kit-industrial): industrial buildings, CC0
- [Kenney City Kit Roads](https://kenney.nl/assets/city-kit-roads): roads and street props, CC0
- [Kenney Factory Kit](https://kenney.nl/assets/factory-kit): machinery and industrial props, CC0
- [Kenney Modular Buildings](https://kenney.nl/assets/modular-buildings): city towers and houses, CC0
- [Kenney Blocky Characters](https://kenney.nl/assets/blocky-characters): animated enemy models, CC0
- [Poly Haven Concrete](https://polyhaven.com/a/concrete): wall texture, CC0
- [Poly Haven Concrete Pavement](https://polyhaven.com/a/concrete_pavement): ground texture, CC0
- [Poly Haven Asphalt 03](https://polyhaven.com/a/asphalt_03): street texture, CC0
- [Poly Haven Factory Brick](https://polyhaven.com/a/factory_brick): warehouse texture, CC0
- [Poly Haven Corrugated Iron](https://polyhaven.com/a/corrugated_iron): industrial structure texture, CC0
- [Three.js](https://threejs.org/): WebGL rendering, MIT

The Kenney license text is stored at `assets/models/License.txt`. Poly Haven assets are published under CC0.

## Check

```bash
./test.sh
```

The check starts the local server when needed and validates the page, scripts, models, textures, and rendering library.

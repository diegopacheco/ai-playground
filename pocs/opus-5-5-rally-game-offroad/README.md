<p align="center"><img src="docs/logo.png" width="220" alt="California Offroad Rally logo"></p>

# California Offroad Rally

A 3D off-road rally game that runs in the browser, raced on **the real ground of four California places**. The terrain comes from real USGS elevation data and the land cover and far scenery come from real Sentinel-2 satellite imagery. You race at:
- Crissy Field under the Golden Gate Bridge.
- Griffith Park below the Hollywood Sign.
- The Lake Tahoe south shore.
- The Yosemite Valley floor.

Pick one of 7 off-road 4x4s modeled on the real vehicles, a paint color and finish, and the weather (clear, rain or snow). Then race 3 laps on a mud track against 3 CPU drivers. The car has an automatic gearbox, a synthesized engine sound, and a co-driver who calls the pace notes.

## How it Works

1. `./play.sh` starts a zero-dependency Node static server and opens the game.
2. `tools/fetch-geo.mjs` downloaded real data for each place into `public/geo/`:
   - AWS Terrain Tiles elevation (USGS 3DEP / SRTM), decoded from PNG in pure Node.
   - Sentinel-2 cloudless satellite tiles.

   The data is committed, so the game runs offline.
3. Each place has two height grids: a fine 4 m grid over 1.6 km for driving, and a coarse 62 m grid over 24 km for the horizon.
4. Tracks are drawn on the real flat ground: Crissy Field, the Griffith Park flats, the Pope Beach shore, and the Yosemite Valley floor. The road is carved into the real terrain with a smoothed profile.
5. Satellite pixels are classified into forest, meadow, dry grass, sand and rock to paint the ground near the track. The same forest mask decides where trees grow, so the forests are where the real forests are.
6. The far terrain is draped with the satellite imagery. Landmarks sit at their real latitude/longitude and size.
7. A fixed 120 Hz simulation steps every car: torque curve, 6-speed automatic gearbox, grip per surface and weather, weight transfer, suspension, airtime and collisions.
8. At speed, the steering asks for a share of the tire grip instead of snapping to the limit. That makes keyboard driving progressive.
9. Pace notes are computed from the track curvature (such as "LEFT 3, 120 m"). They show on the HUD, are spoken by a co-driver voice, and are backed by chevron boards on the corners.
10. A quality governor lowers resolution, shadows and particles until the game holds at least 30 FPS.

## Architecture

![Architecture](docs/architecture.png)

The source is `docs/architecture.svg`.

- `server.js` serves `public/` and `node_modules/three`.
- `public/js/core/` holds pure JavaScript with no DOM and no three.js: geo, tracks, terrain, vehicle physics, pace notes, controls, AI, race, simulation and quality. The browser and the tests run the same modules on the same real terrain.
- `public/js/render/` holds everything three.js: the world, the landmarks, the car models (`cars/` with the builder, wheels and the 7 styles), the showroom, effects and textures.
- `public/js/geoLoader.js` loads the height grids and stitches the satellite tiles.
- `tools/fetch-geo.mjs` is the one-time data download.

## Features

- **Real places**:
  - Crissy Field (San Francisco), with the Golden Gate Bridge at its true position and height (227 m towers), the Marin Headlands, Alcatraz, the Palace of Fine Arts and the downtown skyline with Transamerica and Salesforce Tower.
  - Griffith Park (Los Angeles), with the Hollywood Sign on Mount Lee, the Griffith Observatory and downtown LA.
  - Pope Beach Trail (Lake Tahoe), with the lake at 1897 m, pine forest and Mount Tallac.
  - Yosemite Valley Floor, with the real granite walls of El Capitan and Half Dome from the elevation data, plus Yosemite Falls and Bridalveil Fall.
- **7 recognizable 4x4s**: Jeep Wrangler Rubicon, Ford Bronco Raptor, Toyota Land Cruiser 70, Land Rover Defender 110, Ford F-150 Raptor R, Hummer H1 Alpha and a Baja Trophy Truck. They are built from real side silhouettes with their signature grilles, lights, flares, spares and roof lines, on all-terrain tires with per-car rims.
- **Paint**: 10 colors plus a custom picker, and 6 finishes (Gloss, Metallic, Matte, Pearl, Camo, Carbon).
- **Weather**: clear, rain or snow changes grip, sky, fog, road wetness, snow cover, particles and sound.
- **Muddy track**: rutted mud with normal and roughness maps, puddles that slow you down and splash, tire ruts, mud building up on the paint, and two jumps per stage.
- **Drivable on a keyboard**: progressive steering, consistent grip, pace notes with a red BRAKE warning, a co-driver voice, and chevron boards.
- **3 laps vs 3 CPU drivers**: live standings, lap and best times, minimap, countdown and results.
- **Fully automatic**: a 6-speed automatic gearbox. Hold brake to reverse.
- **Engine noise**: a synthesized V6 or V8 per car, plus tire, gravel, slide, wind, rain and impact sounds.
- **5 cameras and look-back**: chase, far chase, hood, bumper and helicopter. Hold B to look back.
- **30 FPS floor**: 5 quality levels, stepped down automatically.

## Controls

| Key | Action |
|---|---|
| `W` / `Up` | Accelerate |
| `S` / `Down` | Brake, hold to reverse |
| `A` `D` / `Left` `Right` | Steer |
| `Space` | Handbrake |
| `B` | Look back (hold) |
| `C` | Camera: chase, far, hood, bumper, helicopter |
| `R` | Reset to track |
| `M` | Mute sound and co-driver |
| `P` / `Esc` | Pause |
| `H` | Hide or show the controls panel |
| `Enter` | Next step in the menu |

## Stack

- **three.js 0.186**: the only runtime library. It provides WebGL rendering, PBR materials, shadows and the Sky shader.
- **Vanilla ES modules + import map**: no bundler and no framework.
- **Web Audio API and Speech Synthesis**: the engine sound is synthesized and the co-driver uses the browser voice, with no audio files.
- **Canvas 2D**: satellite mosaics, procedural mud, grass, camo, carbon, chevrons, gauge and minimap.
- **Node.js (`node:http`, `node:zlib`)**: the static server and the PNG elevation decoder, with no dependencies.
- **`node --test`**: the built-in test runner.

## Data sources

- Terrain: AWS Terrain Tiles (Mapzen / Tilezen). United States 3DEP (formerly NED) and SRTM terrain data courtesy of the U.S. Geological Survey, and ETOPO1 from NOAA.
- Imagery: Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2016), CC BY 4.0.

The credit line is shown on the in-game HUD. To download the data again, run `node tools/fetch-geo.mjs`.

## Contracts / APIs

There is no backend API. The server only serves static files:

| Path | Served from |
|---|---|
| `GET /`, `GET /js/*`, `GET /geo/<place>/*` | `public/` |
| `GET /vendor/three/*` | `node_modules/three/` |

Paths that resolve outside those folders return `403`. Everything is served with `Cache-Control: no-cache`.

## Key data structures and design decisions

- **Geo grids**: each place has `near.bin` (401x401) and `far.bin` (385x385) as `Int16` quarter-meters, plus a `manifest.json` describing the Web Mercator imagery tiles. Local coordinates are meters east (x) and south (z) of a center lat/lon.
- **Tracks on real ground**: each track is a normalized shape placed by a frame (center, angle, half-length, half-width). Every loop was scored against the real elevation for tightest radius, grade, side slope and water crossings before it was accepted.
- **Terrain**: the real elevation is blended into a smoothed road profile. Physics samples the same triangles the mesh draws. Ground at or below lake level becomes lake bed.
- **Vehicle model**: a bicycle model with world-space velocity. Lateral grip (1.25x grip) is consistent with the yaw limit (1.15x grip), so a car inside its limits turns instead of sliding wide. The steering input scales the yaw limit, so small keyboard taps give small corrections.
- **Pace notes**: corners are grouped by curvature sign. The severity (1 = hairpin to 6 = flat) comes from the tightest radius, and the advised speed is 82% of the grip limit. Distance to the next note is lap-wrap safe.
- **Fixed-step simulation**: `stepSim` runs 120 Hz steps from an accumulator, so physics does not change with frame rate.
- **Two-layer forest**: detailed trees within 180 m of the road (these are the collision obstacles) plus cheap low-poly fill beyond it, both placed by the satellite forest mask.
- **Quality governor**: it drops a level below 36 FPS and climbs back after 4 s above 58 FPS. It skips the first window after a change and doubles its cooldown after every drop.

## How to run

```bash
./play.sh
./stop.sh
```

`play.sh` installs dependencies on first run, starts the server on `http://localhost:7707` and opens the browser.

Tests:

```bash
./scripts/test-all.sh
```

```
ℹ tests 67
ℹ pass 67
ℹ fail 0
```

The tests run on the real terrain data. They cover:
- The real geography: Yosemite floor about 1210 m, Half Dome about 2690 m, the El Capitan rim, Lake Tahoe at 1897 m, Mount Tallac, the Golden Gate strait below sea level, and Mount Lee above the Griffith flats.
- Track geometry: tightest corner over 40 m, no self-overlap, a flat road cross-section, and the road staying above the water.
- A **keyboard driver** with digital keys and a 150 ms reaction delay, following the pace notes on every stage in clear (F-150 Raptor R) and snow (Wrangler). It must stay on the road over 95% of the lap.
- Pace-note direction and severity.
- Progressive steering.
- Physics: gearbox, top speed, reverse, handbrake and jumps.
- Lap rules.
- Full 3-lap CPU races on every stage.
- The quality governor.
- Server path safety.

## Printscreens

The in-race shots were taken in a headless browser with no GPU. That is why the HUD shows the governor at `Potato` holding 30 FPS. On a real GPU it climbs to High or Ultra.

**1. Track selection**: the four real California stages, with the 3D showroom turntable.
![Track selection](printscreens/01-track-select.png)

**2. Jeep Wrangler Rubicon**: the 7-slot grille, round lamps, trapezoid flares, black hardtop and rear spare.
![Wrangler](printscreens/02-vehicle-wrangler.png)

**3. Ford F-150 Raptor R**: the crew cab with the open bed, box flares and running boards.
![Raptor](printscreens/03-vehicle-raptor.png)

**4. Land Rover Defender 110**: the rounded boxy body with the contrast roof and alpine windows.
![Defender](printscreens/04-vehicle-defender.png)

**5. Paint, camo finish**: a procedural camouflage pattern on the body.
![Camo paint](printscreens/05-paint-camo.png)

**6. Conditions**: clear, rain or snow, with the race summary.
![Conditions](printscreens/06-conditions.png)

**7. San Francisco from the helicopter cam (looking back)**: the Golden Gate Bridge at its real position across the bay, the Marin Headlands behind it, and the Presidio cypress and Crissy Field below. The pace note calls "LEFT 3" and chevrons mark the corner.
![Golden Gate](printscreens/07-sf-golden-gate.png)

**8. San Francisco, look back (B)**: the Wrangler's front with the bridge towers and the start arch behind.
![SF look back](printscreens/08-sf-look-back.png)

**9. Los Angeles, Griffith Park**: dry golden grass, oaks and palms on the real Griffith flats, with the real hills on the horizon.
![Griffith Park](printscreens/09-la-griffith.png)

**10. Lake Tahoe, look back**: the Defender (the DEFENDER grille) in the south-shore pine forest.
![Tahoe Defender](printscreens/10-tahoe-defender.png)

**11. Lake Tahoe from the helicopter**: dense pine forest placed from the satellite forest mask, with the glare of the lake on the horizon.
![Tahoe lake](printscreens/11-tahoe-lake.png)

**12. Yosemite Valley**: the Bronco under the real granite valley wall, among the ponderosa pines of the valley floor.
![Yosemite Valley](printscreens/12-yosemite-valley.png)

**13. Yosemite from the helicopter**: the track winding through the valley-floor forest past the start arch.
![Yosemite helicopter](printscreens/13-yosemite-heli.png)

**14. Yosemite in snow**: the Hummer H1 on a snow-covered floor among snowy pines.
![Yosemite snow](printscreens/14-yosemite-snow.png)

**15. Lake Tahoe in rain**: the Land Cruiser in a downpour under an overcast sky, on a soaked road.
![Tahoe rain](printscreens/15-tahoe-rain.png)

**16. San Francisco in rain**: the Baja Trophy Truck with its headlights on in the storm, from the far chase cam.
![SF rain](printscreens/16-sf-rain.png)

**17. Hood cam**: the view over the hood, following the rivals into the Tahoe pines.
![Hood cam](printscreens/17-hood-cam.png)

**18. Pause**: resume, restart or main menu.
![Pause](printscreens/18-pause.png)

**19. Results**: position, driver, vehicle, total time and best lap. This shot uses a sample race state rendered by the real `showResults`, because a full 3-lap race takes a few minutes.
![Results](printscreens/19-results.png)

## Scripts

All scripts live in `scripts/` and run from any directory of the repository.

| Script | What it does |
|---|---|
| `./play.sh` | Starts the game and opens it in the browser |
| `./stop.sh` | Stops the game |
| `./scripts/setup.sh` | Installs dependencies and prepares the app |
| `./scripts/start-all.sh` | Starts every service and prints the full link of each one |
| `./scripts/status.sh` | Shows every service port as UP or DOWN |
| `./scripts/test-all.sh` | Runs every test suite |
| `./scripts/ui.sh` | Opens the UI in the browser |
| `./scripts/stop-all.sh` | Stops every service |

Ports are declared in `scripts/ports.env` (`game=7707`).

```bash
./scripts/setup.sh
./scripts/start-all.sh
./scripts/status.sh
./scripts/ui.sh
./scripts/stop-all.sh
```

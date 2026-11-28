# Changelog

Records what changed for the player and the operator of machine-claw-game.

## 0.2.0 - 2026-11-27

### Added
- The machine holds 40 stuffies, 2.5x the old 16, stacked on a layered grid so they settle without exploding at 120 steps per second.
- Six new plush kinds: penguin, octopus, dino, unicorn, fox and whale; every one of the fourteen kinds is always in the pile.
- MY STUFFIES button with the total won; it opens a pastel 3D showcase of every kind won with its name and count, drag or swipe to spin, × or `Esc` to close.
- Wins are saved in localStorage and survive reloads; broken or blocked storage is ignored.

### Changed
- Stuffies are cuter: bigger glossy eyes with two highlights and blush cheeks.
- Toys are a little smaller (radius 7.8 to 10 cm) so the bigger pile fits under the claw.
- Each plush kind is built once as merged geometry and reused, so 40 toys draw fast on phones.
- The machine pauses while the showcase is open.

### Verified
- `bun test`: 27 passed, 0 failed, 0 skipped. `bun run build`: typecheck clean, build succeeded.
- Headless aim at the top toy of the 40 toy pile won 11 of 16 tries (before: 17 of 24 on 16 toys).
- Headless Chromium on the GPU held 60 fps on the machine and the showcase at 1280x800 and 390x844, also with 4x CPU throttling.
- At 390x844 no HUD or showcase element overflows the screen, a touch swipe spins the showcase and tapping × closes it.

## 0.1.0 - 2026-11-20

### Added
- A 3D claw machine cabinet with a pile of 16 plush toys in eight kinds: bear, bunny, cat, panda, frog, chick, koala and pig.
- COIN button and `C` key insert a coin; one coin is one try with a 20 second timer and extra coins are kept as credits, up to 9.
- Joystick, arrow keys and `WASD` move the claw; the DROP button and `Space` drop it, and it also drops by itself when the timer ends.
- The claw closes, lifts and carries toys by physics contact and friction only, so a toy can slip out on the way to the chute.
- A toy that falls through the chute adds to PRIZES, shows on the prize shelf, fires confetti and is replaced by a new toy falling from the top.
- VIEW button and `V` key switch between the front camera and a side camera.
- Touch layout for phones in portrait and landscape.
- Sound effects for coin, start, drop, grab, win, miss and the claw motor.
- Scripts `setup.sh`, `start-all.sh`, `status.sh`, `test-all.sh`, `ui.sh` and `stop-all.sh`.

### Changed
- The claw motor now makes a soft whir that glides up when the claw starts moving, instead of a harsh buzz; the drop, grab and miss sounds are softer too.
- The empty dark room is now an arcade hall with neon signs, rows of game cabinets, a prize wall, string lights and a patterned carpet.
- The camera can be moved: drag to orbit around the machine, scroll or pinch to zoom; VIEW still jumps between front and side.
- The machine has rounded edges, glossy gradient paint, chrome trims, wider LED strips, softer shadows and a metal-lined chute.
- The game now advances at a fixed 120 steps per second, so motion and grabs are the same on every screen.

### Fixed
- The LUCKY CLAW sign was cut off on both sides because the text was wider than the sign; the text is now sized to fit.
- The chute looked like a black box sitting on the floor because its walls were unlit black; they are now brushed metal and it reads as a hole.

### Verified
- `./scripts/test-all.sh` passed: typecheck clean, 15 tests passed with 0 failed and 0 skipped, production build succeeded.
- A headless full round carried a toy from the pile area to the chute and counted 1 prize; dropping on an empty spot counted 0.
- Measured on a single toy over 8 tries each: perfect aim won 8 of 8, 6 cm off won 4 of 8. Perfect aim on the top toy of a full pile won 17 of 24.
- In headless Chromium at 1280x800 a round played with the keyboard won a frog on the first aimed try and the PRIZES counter went from 0 to 1.
- At 390x844 the COIN button, the joystick drag and the DROP button moved the game from INSERT COIN to aiming to dropping.
- After the arcade hall, camera and 120 steps per second changes, a keyboard round in headless Chromium at 1280x800 again won a stuffy on the first aimed try, and a mouse drag plus scroll orbited and zoomed the camera.
- `start-all.sh`, `status.sh` and `stop-all.sh` ran from other directories; running `stop-all.sh` twice exited 0 and the port ended DOWN.

# Changelog

Records what changed for the player and the operator of machine-claw-game.

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

### Verified
- `./scripts/test-all.sh` passed: typecheck clean, 15 tests passed with 0 failed and 0 skipped, production build succeeded.
- A headless full round carried a toy from the pile area to the chute and counted 1 prize; dropping on an empty spot counted 0.
- Measured on a single toy over 8 tries each: perfect aim won 8 of 8, 6 cm off won 4 of 8. Perfect aim on the top toy of a full pile won 17 of 24.
- In headless Chromium at 1280x800 a round played with the keyboard won a frog on the first aimed try and the PRIZES counter went from 0 to 1.
- At 390x844 the COIN button, the joystick drag and the DROP button moved the game from INSERT COIN to aiming to dropping.
- `start-all.sh`, `status.sh` and `stop-all.sh` ran from other directories; running `stop-all.sh` twice exited 0 and the port ended DOWN.

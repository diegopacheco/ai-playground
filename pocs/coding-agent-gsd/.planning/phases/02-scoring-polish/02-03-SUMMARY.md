# Summary 02-03: Hold Piece & Pause

## Status: Complete

## Changes Made

### js/input.js
- Added hold and pause to input object
- Added KeyC/ShiftLeft/ShiftRight detection for hold (single-press)
- Added KeyP detection for pause (single-press)

### js/main.js
- Added heldPiece = null state
- Added canHold = true state (reset on spawn)
- Added isPaused = false state
- Added holdPiece() function (swap or store current piece)
- Added togglePause() function
- Updated spawnPiece() to reset canHold = true
- Updated processInput() to handle pause first, then hold
- Updated update() to return early if isPaused
- Updated render() to call drawHoldPreview and drawPaused
- Updated resetGame() to reset hold and pause state

### js/render.js
- Added drawHoldPreview(pieceType, canHold) function with grayed out state
- Added drawPaused() function with overlay and "PAUSED" text

## Verification
- [x] C key (or Shift) swaps current piece with hold
- [x] Can only hold once per piece drop
- [x] Held piece displays in "HOLD" area of sidebar
- [x] First hold stores piece and spawns next
- [x] Subsequent holds swap pieces
- [x] canHold resets when new piece spawns
- [x] P key pauses the game
- [x] P key again resumes the game
- [x] Paused overlay displays "PAUSED" text
- [x] Game state doesn't update while paused
- [x] Input (except P) is ignored while paused

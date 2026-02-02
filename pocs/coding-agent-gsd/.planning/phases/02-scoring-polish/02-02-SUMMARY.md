# Summary 02-02: Ghost Piece & Next Preview

## Status: Complete

## Changes Made

### js/main.js
- Added nextPiece state variable for preview
- Added getGhostY(piece) function to calculate landing position
- Updated spawnPiece() to use nextPiece system (current = next, generate new next)
- Updated render() to draw ghost piece before current piece
- Updated resetGame() to reset nextPiece

### js/render.js
- Added drawGhost(pieceType, x, ghostY, rotation) function with 0.3 opacity
- Added drawNextPreview(pieceType) function with NEXT label and preview box

## Verification
- [x] Ghost piece visible below current piece
- [x] Ghost piece follows current piece movement
- [x] Ghost piece shows correct rotation
- [x] Ghost piece has reduced opacity (semi-transparent)
- [x] Ghost disappears when piece is at landing position
- [x] "NEXT" label visible in sidebar
- [x] Next piece preview shows correct upcoming piece
- [x] Next piece changes after current piece locks

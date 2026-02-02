# Summary 02-01: UI Layout & Score Display

## Status: Complete

## Changes Made

### js/render.js
- Added SIDEBAR_WIDTH = 120 constant
- Updated setupCanvas() to create canvas of (COLS * CELL_SIZE + SIDEBAR_WIDTH) x (ROWS * CELL_SIZE)
- Added drawSidebar() function for dark sidebar background
- Added drawScore(score, level) function with styled labels and values

### js/main.js
- Added level = 1 state variable
- Added calculateLevel() function: Math.floor(score / 100) + 1
- Added checkLevelUp() function to detect level changes
- Added onLevelUp() function (placeholder logs "Level up to X")
- Updated update() to call checkLevelUp() after score increases
- Updated render() to call drawSidebar() and drawScore()
- Updated resetGame() to reset level

## Verification
- [x] Canvas displays at 420x600 (300 board + 120 sidebar)
- [x] Sidebar has distinct background from board
- [x] Score displays and updates when rows clear
- [x] Level displays starting at 1
- [x] Level increases when score reaches 100, 200, etc.
- [x] Console logs "Level up" when level changes
- [x] Board area rendering unchanged

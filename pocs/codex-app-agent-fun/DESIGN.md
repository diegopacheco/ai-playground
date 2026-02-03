# Tetris Web App Design

## Overview
This app is a classic Tetris game with a React 19 user interface and a Bun server that serves static assets. It includes an admin panel for runtime configuration of themes, timing, difficulty, and additional gameplay options.

## Goals
- Provide a playable Tetris experience with keyboard controls
- Enforce the scoring and timing rules from the prompt
- Allow admin configuration of core gameplay and UI settings
- Keep the stack minimal and easy to run

## Core Rules
- Each cleared line grants 10 points by default
- Every 50 points advances one level
- Every 30 seconds there is a 50% chance the UI freezes for 10 seconds
- Every 50 seconds the UI grows larger

## Architecture
- Bun server in `src/index.ts` serves files from `public/`
- Client uses React 19 loaded from ESM CDN
- All game logic runs in the browser

## UI
- Main board rendered as a grid
- Side panel for score, level, status, and next piece
- Admin panel for settings
- Freeze overlay shown when the UI is frozen

## Admin Configuration
- Theme selection (Classic, Neon, Light)
- Difficulty (Easy, Normal, Hard, Custom)
- Base and minimum drop times
- Board width and height
- Points per line and points per level
- Freeze interval, duration, and chance
- UI grow interval and step size
- Optional ghost piece and next piece visibility

## Game Logic
- Pieces spawn at the top center and fall at a rate derived from difficulty and level
- Rotation uses a kick list to shift left or right when blocked
- Lines are cleared on lock and scoring is applied immediately
- Level is derived from total score and points per level

## Timing Model
- Drop timer updates based on level and configured minimum drop time
- Freeze timer runs every 30 seconds and triggers with 50% chance
- Grow timer runs every 50 seconds and increases UI scale by a configured step

## Data Model
- Grid is a 2D array of color values or null
- Active piece includes shape matrix and color
- Settings stored in React state and applied on demand

## Controls
- Arrow keys move and rotate the piece
- Space triggers hard drop
- P toggles pause

## Running
- `run.sh` starts the Bun server

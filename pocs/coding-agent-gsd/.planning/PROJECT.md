# Tetris Twist

## What This Is

A browser-based Tetris game with unique mechanics: alternating 10-second play/freeze cycles, a board that grows over time, and real-time admin controls. Two UIs — player plays in one tab, admin tweaks configs in another, changes apply instantly.

## Core Value

The real-time admin control loop — admin changes themes, speed, and scoring while the player experiences those changes live. Without this sync, it's just another Tetris clone.

## Requirements

### Validated (v1.0 - Shipped 2026-02-03)

- [x] Classic Tetris gameplay (falling pieces, rotation, movement)
- [x] 10-second play / 10-second freeze cycle
- [x] Board grows (wider and taller) every 30 seconds
- [x] Clearing a row awards points (default 10)
- [x] 100 points advances to next level
- [x] Level up triggers visual theme change
- [x] Pre-built themes with colors and piece shapes (3 themes: Classic, Neon, Retro)
- [x] Admin UI in separate browser tab
- [x] Admin can switch themes (applied real-time)
- [x] Admin can change fall speed (applied real-time)
- [x] Admin can change points per row (applied real-time)
- [x] Admin can change board growth rate (applied real-time)
- [x] Player and Admin sync via same browser (tabs)

### Active

(None - v1.0 complete, ready for v2 milestone)

### Out of Scope

- Sound/music — not requested
- Multiplayer/networking — same browser only
- Mobile app — web browser target
- User accounts/persistence — session-only gameplay
- Leaderboards — not requested

## Context

**v1.0 Shipped:** 2026-02-03

This is a twist on classic Tetris designed to showcase real-time configuration. The freeze mechanic creates tension (you can see pieces but can't act), and the growing board adds long-game strategy. The admin panel makes it feel like a live broadcast where someone controls the experience.

The same-browser constraint simplifies architecture — uses BroadcastChannel API for real-time sync between tabs.

## Constraints

- **Platform**: Web browser (HTML/CSS/JS)
- **No frameworks**: Keep dependencies minimal per project guidelines
- **Same browser**: Admin and player tabs on same machine

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| BroadcastChannel for sync | Native browser API, no server needed, real-time | Validated - instant sync works well |
| Pre-built themes only | Simpler than theme editor, faster to ship | Validated - 3 themes sufficient |
| No sound | Not requested, reduces scope | Validated - not needed |
| Vanilla JS + Canvas | No dependencies constraint | Validated - clean, fast |
| GameState enum | More scalable state management than booleans | Validated - cleaner code |
| Board grows at bottom | Preserves existing piece positions | Validated - seamless growth |
| MAX_ROWS = 30 | 50% growth limit from initial 20 | Validated - good balance |

---
*Last updated: 2026-02-03 after v1.0 milestone shipped*

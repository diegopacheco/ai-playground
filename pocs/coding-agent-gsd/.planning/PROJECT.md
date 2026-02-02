# Tetris Twist

## What This Is

A browser-based Tetris game with unique mechanics: alternating 10-second play/freeze cycles, a board that grows over time, and real-time admin controls. Two UIs — player plays in one tab, admin tweaks configs in another, changes apply instantly.

## Core Value

The real-time admin control loop — admin changes themes, speed, and scoring while the player experiences those changes live. Without this sync, it's just another Tetris clone.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Classic Tetris gameplay (falling pieces, rotation, movement)
- [ ] 10-second play / 10-second freeze cycle
- [ ] Board grows (wider and taller) every 30 seconds
- [ ] Clearing a row awards points (default 10)
- [ ] 100 points advances to next level
- [ ] Level up triggers visual theme change
- [ ] Pre-built themes with colors and piece shapes
- [ ] Admin UI in separate browser tab
- [ ] Admin can switch themes (applied real-time)
- [ ] Admin can change fall speed (applied real-time)
- [ ] Admin can change points per row (applied real-time)
- [ ] Admin can change board growth rate (applied real-time)
- [ ] Player and Admin sync via same browser (tabs)

### Out of Scope

- Sound/music — not requested
- Multiplayer/networking — same browser only
- Mobile app — web browser target
- User accounts/persistence — session-only gameplay
- Leaderboards — not requested

## Context

This is a twist on classic Tetris designed to showcase real-time configuration. The freeze mechanic creates tension (you can see pieces but can't act), and the growing board adds long-game strategy. The admin panel makes it feel like a live broadcast where someone controls the experience.

The same-browser constraint simplifies architecture — can use BroadcastChannel API or localStorage events for real-time sync between tabs.

## Constraints

- **Platform**: Web browser (HTML/CSS/JS)
- **No frameworks**: Keep dependencies minimal per project guidelines
- **Same browser**: Admin and player tabs on same machine

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| BroadcastChannel for sync | Native browser API, no server needed, real-time | — Pending |
| Pre-built themes only | Simpler than theme editor, faster to ship | — Pending |
| No sound | Not requested, reduces scope | — Pending |

---
*Last updated: 2026-02-01 after initialization*

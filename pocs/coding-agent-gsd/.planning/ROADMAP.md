# Roadmap: Tetris Twist

**Created:** 2026-02-02
**Depth:** Quick (4 phases)
**Core Value:** Real-time admin control loop

## Overview

| Phase | Name | Goal | Requirements | Status |
|-------|------|------|--------------|--------|
| 1 | Core Engine | Playable Tetris with all mechanics | 15 | ✓ Complete |
| 2 | Scoring & Polish | Complete game feel with scoring, preview, hold | 9 | ✓ Complete |
| 3 | Themes & Admin | Admin panel with real-time control | 12 | ✓ Complete |
| 4 | Unique Mechanics | Freeze cycle and board growth | 5 | ○ Pending |

**Total:** 4 phases | 37 requirements | 77% complete

---

## Phase 1: Core Engine

**Goal:** Playable Tetris — pieces fall, player controls them, lines clear, game ends when stacked.

**Requirements:**
- CORE-01: Game displays 10-column, 20-row grid on canvas
- CORE-02: 7 standard tetrominoes spawn
- CORE-03: Pieces fall automatically at configurable speed
- CORE-04: Player can move piece left/right
- CORE-05: Player can soft drop
- CORE-06: Player can hard drop
- CORE-07: Player can rotate piece
- CORE-08: Rotation uses wall kicks
- CORE-09: Pieces lock when they land
- CORE-10: Completed rows clear
- CORE-11: Game ends when pieces stack to top
- TECH-01: HTML5 Canvas for rendering
- TECH-02: 60fps with requestAnimationFrame
- TECH-04: No external dependencies
- TECH-05: Canvas renders crisp on high-DPI

**Success Criteria:**
1. Player can start game and see empty grid
2. Pieces spawn at top and fall automatically
3. Arrow keys move and rotate pieces
4. Space bar hard drops piece
5. Full rows disappear when completed
6. Game shows "Game Over" when board fills

---

## Phase 2: Scoring & Polish

**Goal:** Complete game feel — score, levels, next preview, hold, ghost, pause.

**Requirements:**
- EHNC-01: Ghost piece shows landing position
- EHNC-02: Hold piece (swap once per drop)
- EHNC-03: Next piece preview
- EHNC-04: Pause/resume with P key
- SCOR-01: Score display updates real-time
- SCOR-02: Clearing row awards points
- SCOR-03: Level display
- SCOR-04: 100 points → next level
- SCOR-05: Level up triggers theme change

**Success Criteria:**
1. Ghost piece visible below current piece
2. Pressing hold key swaps piece to hold slot
3. Next piece visible in preview area
4. P key pauses game, shows paused state
5. Score increases when rows clear
6. Level increases at 100 points
7. Visual change occurs on level up

---

## Phase 3: Themes & Admin

**Goal:** Admin panel controls game in real-time — themes, speed, scoring all adjustable live.

**Requirements:**
- THEM-01: 3 themes (Classic, Neon, Retro)
- THEM-02: Themes define colors and piece shapes
- THEM-03: Theme changes apply instantly
- THEM-04: Level up cycles through themes
- ADMN-01: Admin panel in separate tab
- ADMN-02: Select active theme
- ADMN-03: Adjust fall speed
- ADMN-04: Adjust points per row
- ADMN-05: Adjust board growth interval
- ADMN-06: See live game stats
- ADMN-07: All changes sync real-time
- TECH-03: BroadcastChannel for sync

**Plans:** 4 plans (all complete)

Plans:
- [x] 03-01-PLAN.md — Theme system foundation (themes.js, board refactor)
- [x] 03-02-PLAN.md — Theme-aware rendering (render.js refactor)
- [x] 03-03-PLAN.md — Sync infrastructure (BroadcastChannel, theme cycling)
- [x] 03-04-PLAN.md — Admin panel (admin.html, controls, live stats)

**Success Criteria:**
1. Admin panel opens in new tab
2. Changing theme in admin instantly updates game colors
3. Adjusting speed slider changes how fast pieces fall
4. Changing points per row affects scoring immediately
5. Admin panel shows current score and level live
6. Game continues working if admin tab closes

---

## Phase 4: Unique Mechanics

**Goal:** Implement signature mechanics — freeze cycles and growing board.

**Requirements:**
- UNIQ-01: 10s play / 10s freeze cycles
- UNIQ-02: Freeze state with visual indicator and countdown
- UNIQ-03: Board grows every 30s
- UNIQ-04: Growth is smooth, pieces stay in place
- UNIQ-05: Board has maximum size limit

**Plans:** 2 plans

Plans:
- [ ] 04-01-PLAN.md — Freeze cycle mechanics (state machine, overlay, countdown)
- [ ] 04-02-PLAN.md — Board growth mechanics (dynamic height, canvas resize, max limit)

**Success Criteria:**
1. Game alternates between play and freeze states
2. Freeze state shows countdown timer and grayed pieces
3. Board visibly expands after 30 seconds
4. Existing pieces remain in correct positions after growth
5. Board stops growing at maximum size
6. Admin can adjust growth interval and it takes effect

---

## Phase Dependencies

```
Phase 1 (Core Engine)
    ↓
Phase 2 (Scoring & Polish)
    ↓
Phase 3 (Themes & Admin)
    ↓
Phase 4 (Unique Mechanics)
```

All phases are sequential — each builds on the previous.

---

## Milestone: v1.0

**Definition of Done:**
- All 37 v1 requirements complete
- Game playable end-to-end
- Admin panel functional
- All 3 themes working
- Freeze and growth mechanics active

---
*Roadmap created: 2026-02-02*
*Last updated: 2026-02-02 after Phase 4 planning*

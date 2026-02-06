# Roadmap: Tetris Twist

**Created:** 2026-02-02
**Depth:** Quick
**Core Value:** Real-time admin control loop

## Overview

| Phase | Name | Goal | Requirements | Status |
|-------|------|------|--------------|--------|
| 1 | Core Engine | Playable Tetris with all mechanics | 15 | ✓ Complete |
| 2 | Scoring & Polish | Complete game feel with scoring, preview, hold | 9 | ✓ Complete |
| 3 | Themes & Admin | Admin panel with real-time control | 12 | ✓ Complete |
| 4 | Unique Mechanics | Freeze cycle and board growth | 5 | ✓ Complete |
| 5 | Additional Themes | Visual variety with new themes | 3 | ✓ Complete |
| 6 | Session Statistics | Performance tracking and display | 4 | ✓ Complete |
| 7 | Combo System | Reward consecutive line clears | 5 | ✓ Complete |
| 8 | T-Spin Detection | Advanced scoring for skilled play | 5 | ✓ Complete |
| 9 | Audio Feedback | Sound effects for game events | 5 | ✓ Complete |
| 10 | Keyboard Remapping | Customizable controls | 5 | ✓ Complete |
| 11 | Combo Pitch Scaling | Escalating pitch feedback for combos | 5 | Pending |
| 12 | Personal Best Tracking | Cross-session record tracking | 5 | Pending |
| 13 | Key Binding Export/Import | Share and backup key configurations | 5 | Pending |
| 14 | Background Music | Procedural atmospheric music | 5 | Pending |

**Milestone v1.0:** 4 phases | 37 requirements | 100% complete
**Milestone v2.0:** 6 phases | 27 requirements | 100% complete
**Milestone v3.0:** 4 phases | 20 requirements | 0% complete

---

## Milestone: v1.0 (SHIPPED)

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

**Plans:** 2 plans (all complete)

Plans:
- [x] 04-01-PLAN.md — Freeze cycle mechanics (state machine, overlay, countdown)
- [x] 04-02-PLAN.md — Board growth mechanics (dynamic height, canvas resize, max limit)

**Success Criteria:**
1. Game alternates between play and freeze states
2. Freeze state shows countdown timer and grayed pieces
3. Board visibly expands after 30 seconds
4. Existing pieces remain in correct positions after growth
5. Board stops growing at maximum size
6. Admin can adjust growth interval and it takes effect

---

## Milestone: v2.0 (SHIPPED)

---

## Phase 5: Additional Themes

**Goal:** Players experience visual variety with new accessible themes.

**Dependencies:** None (extends existing theme system)

**Requirements:**
- THEM-05: Add Minimalist theme (clean, simple colors)
- THEM-06: Add High Contrast theme (accessibility-focused)
- THEM-07: Admin theme selector shows all 5+ themes

**Plans:** 1 plan (complete)

Plans:
- [x] 05-01-PLAN.md — Add Minimalist and High Contrast themes

**Success Criteria:**
1. Player can select Minimalist theme and see clean color palette
2. Player can select High Contrast theme with strong color differentiation
3. Admin panel theme dropdown lists 5 themes
4. Theme changes during gameplay apply instantly without visual glitches

---

## Phase 6: Session Statistics

**Goal:** Players see detailed performance metrics during and after gameplay.

**Dependencies:** None (pure tracking, no game mechanic changes)

**Requirements:**
- STAT-01: Track basic stats (score, lines, level, time, pieces placed)
- STAT-02: Track advanced stats (PPS, APM, efficiency, tetris rate)
- STAT-03: Display real-time stats in sidebar
- STAT-04: Session summary screen shows all stats on game over

**Plans:** 2 plans (all complete)

Plans:
- [x] 06-01-PLAN.md — Stats tracking module and basic sidebar display
- [x] 06-02-PLAN.md — Advanced stats calculation and game over summary screen

**Success Criteria:**
1. Player sees live stats sidebar with score, lines, level, time, pieces
2. Sidebar displays real-time PPS and APM calculations
3. Game over screen shows complete session summary with all metrics
4. Stats update correctly during freeze cycles (time continues, actions pause)

---

## Phase 7: Combo System

**Goal:** Players receive bonus points for consecutive line clears.

**Dependencies:** Phase 6 (stats tracking for combo display)

**Requirements:**
- COMB-01: Combo counter tracks consecutive line clears
- COMB-02: Combo resets when piece locks without clearing lines
- COMB-03: Combo awards bonus points (50 x combo x level)
- COMB-04: Visual combo counter displays during active combo
- COMB-05: Back-to-Back bonus (1.5x) for consecutive Tetris/T-spin clears

**Plans:** 2 plans (all complete)

Plans:
- [x] 07-01-PLAN.md — Core combo and B2B scoring logic in main.js
- [x] 07-02-PLAN.md — Visual combo display and statistics tracking

**Success Criteria:**
1. Player clears line, combo counter shows "1x Combo"
2. Player clears another line immediately, combo increments to "2x Combo"
3. Player locks piece without clearing, combo resets to zero
4. Combo bonus points add to score after each clear
5. Consecutive Tetris clears show "Back-to-Back" indicator and award 1.5x points

---

## Phase 8: T-Spin Detection

**Goal:** Players receive recognition and bonus points for advanced T-spin moves.

**Dependencies:** Phase 7 (combo system for B2B and scoring integration)

**Requirements:**
- TSPN-01: Game detects T-spin when T-piece locks after rotation with 3+ corners occupied
- TSPN-02: Mini T-spin detected when only 1 front corner occupied
- TSPN-03: Full T-spin detected when 2 front corners occupied
- TSPN-04: T-spin awards bonus points (Guideline: mini 100/200/400, full 400/800/1200/1600 × level)
- TSPN-05: Visual indicator displays T-spin type on detection

**Plans:** 2 plans (all complete)

Plans:
- [x] 08-01-PLAN.md — T-spin detection logic and scoring integration
- [x] 08-02-PLAN.md — Visual T-spin indicator and session summary

**Success Criteria:**
1. Player rotates T-piece into tight space, locks it, and sees "T-Spin Mini" indicator
2. Player performs full T-spin, sees "T-Spin" indicator with bonus points
3. T-spin with line clear awards base T-spin points plus line clear bonus
4. Back-to-Back T-spin clears show B2B indicator and multiply points by 1.5x
5. T-spin detection only triggers after rotation, not simple drops

---

## Phase 9: Audio Feedback

**Goal:** Players receive auditory feedback for game events.

**Dependencies:** None (independent feature)

**Requirements:**
- AUDIO-01: Sound effect plays on piece land
- AUDIO-02: Sound effect plays on line clear
- AUDIO-03: Sound effect plays on Tetris (4-line clear)
- AUDIO-04: Sound effect plays on game over
- AUDIO-05: Mute toggle in admin panel persists to localStorage

**Plans:** 2 plans (all complete)

Plans:
- [x] 09-01-PLAN.md — Audio module with OscillatorNode sound effects
- [x] 09-02-PLAN.md — Wire audio triggers and admin mute toggle

**Success Criteria:**
1. Player hears distinct sound when piece locks on surface
2. Line clear plays satisfying clear sound
3. Tetris clear plays special higher-pitched sound
4. Game over plays final sound effect
5. Admin can toggle mute and setting persists across browser sessions

---

## Phase 10: Keyboard Remapping

**Goal:** Players customize all controls to their preferences.

**Dependencies:** None (refactors existing input system)

**Requirements:**
- KEYS-01: All game controls are remappable
- KEYS-02: Visual settings UI for key binding
- KEYS-03: Key bindings persist to localStorage
- KEYS-04: Conflict detection prevents duplicate bindings
- KEYS-05: Default bindings restore option

**Plans:** 2 plans (all complete)

Plans:
- [x] 10-01-PLAN.md — Keymap system with localStorage persistence
- [x] 10-02-PLAN.md — Key binding UI in admin panel

**Success Criteria:**
1. Player opens settings UI and sees current key bindings
2. Player clicks a control, presses new key, binding updates immediately
3. Player attempts to bind same key to two controls, receives conflict warning
4. Player closes game, reopens, and custom bindings are still active
5. Player clicks "Restore Defaults" and all controls return to arrow keys and spacebar

---

## Milestone: v3.0 (IN PROGRESS)

---

## Phase 11: Combo Pitch Scaling

**Goal:** Players hear escalating pitch feedback that rewards combo chains.

**Dependencies:** Phase 9 (audio module), Phase 7 (combo system)

**Requirements:**
- PITCH-01: Line clear sound pitch increases proportionally with combo counter
- PITCH-02: Pitch scaling uses smooth parameter ramping to prevent clicks
- PITCH-03: Pitch capped at 10x combo to prevent painful frequencies
- PITCH-04: Different pitch patterns for single/double/triple/Tetris clears
- PITCH-05: Pitch resets to base when combo breaks

**Plans:** 1 plan

Plans:
- [ ] 11-01-PLAN.md — Pitch scaling in audio.js and wiring in main.js

**Success Criteria:**
1. Player clears line with combo counter at 3x, hears noticeably higher-pitched sound
2. Pitch increases smoothly without audio clicks or pops
3. Pitch stops increasing after 10x combo even if combo continues
4. Tetris clear sounds different from single line clear at same combo level
5. Combo breaks, next line clear plays at base pitch again

---

## Phase 12: Personal Best Tracking

**Goal:** Players track and beat personal records across sessions.

**Dependencies:** Phase 6 (session statistics)

**Requirements:**
- BEST-01: Track high score, lines, level across sessions in localStorage
- BEST-02: Display personal best comparison at game over screen
- BEST-03: Show "New Record" visual notification when beating personal best
- BEST-04: Track timestamp of when personal best was achieved
- BEST-05: Clear/reset personal bests option in admin panel

**Success Criteria:**
1. Player beats high score, sees "New Record" celebration on game over screen
2. Game over screen shows previous best score and current score side-by-side
3. Personal bests persist after closing and reopening browser
4. Admin panel shows "Clear Records" button that resets all personal bests
5. Personal best timestamp shows when record was achieved

---

## Phase 13: Key Binding Export/Import

**Goal:** Players share and backup custom key configurations.

**Dependencies:** Phase 10 (keyboard remapping)

**Requirements:**
- EXPORT-01: Export current key bindings to tetris_keybindings.json file
- EXPORT-02: Import key bindings from JSON file with validation
- EXPORT-03: Show error messages for invalid import files
- EXPORT-04: Export includes metadata (date, game version)
- EXPORT-05: Export/Import buttons in admin panel Controls section

**Success Criteria:**
1. Player clicks Export button in admin, downloads tetris_keybindings.json file
2. Player imports valid JSON file, custom bindings apply immediately
3. Player imports corrupted file, sees clear error message
4. Exported file contains date and version metadata
5. Export/Import buttons visible in Controls section of admin panel

---

## Phase 14: Background Music

**Goal:** Players experience atmospheric procedural music during gameplay.

**Dependencies:** Phase 9 (audio module)

**Requirements:**
- MUSIC-01: Procedural background music via continuous OscillatorNode
- MUSIC-02: Seamless looping without restart gaps
- MUSIC-03: Separate mute toggle from sound effects
- MUSIC-04: Music pauses when game pauses, stops on game over
- MUSIC-05: Music gain lower than effects to not overpower gameplay

**Success Criteria:**
1. Player starts game, hears continuous background music playing
2. Music loops seamlessly without noticeable gaps or restarts
3. Player mutes music via toggle, sound effects still play
4. Player pauses game, music pauses; resumes game, music continues
5. Music volume lower than sound effects, does not interfere with gameplay

---

## Phase Dependencies

```
Milestone v1.0 (Complete)
    ↓
Phase 5 (Additional Themes) ← independent
Phase 6 (Session Statistics) ← independent
    ↓
Phase 7 (Combo System)
    ↓
Phase 8 (T-Spin Detection)

Phase 9 (Audio Feedback) ← independent
    ↓
Phase 11 (Combo Pitch Scaling)
Phase 14 (Background Music)

Phase 10 (Keyboard Remapping) ← independent
    ↓
Phase 13 (Key Binding Export/Import)

Phase 6 (Session Statistics)
    ↓
Phase 12 (Personal Best Tracking) ← independent
```

**v1.0 Sequential:** Phase 1 → 2 → 3 → 4
**v2.0 Parallel branches:** Phase 5, Phase 6→7→8, Phase 9, Phase 10
**v3.0 Parallel branches:** Phase 11 (after 9), Phase 12 (after 6), Phase 13 (after 10), Phase 14 (after 9)

---

## Milestone: v2.0 Definition of Done

- All 27 v2 requirements complete
- 2 new themes working (Minimalist, High Contrast)
- Session statistics display and summary functional
- Combo system awards points correctly
- T-spin detection with visual feedback
- Sound effects for all major events
- Full keyboard remapping with conflict detection

---

## Milestone: v3.0 Definition of Done

- All 20 v3 requirements complete
- Combo pitch scaling enhances audio feedback
- Personal bests tracked across browser sessions
- Key bindings exportable and importable via JSON
- Procedural background music playing during gameplay
- Zero new external dependencies

---
*Roadmap created: 2026-02-02*
*Last updated: 2026-02-06 with v3.0 milestone phases 11-14*

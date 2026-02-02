# Requirements: Tetris Twist

**Defined:** 2026-02-02
**Core Value:** Real-time admin control loop — admin tweaks, player experiences instantly

## v1 Requirements

### Core Mechanics

- [ ] **CORE-01**: Game displays 10-column, 20-row grid on canvas
- [ ] **CORE-02**: 7 standard tetrominoes spawn (I, O, T, S, Z, J, L)
- [ ] **CORE-03**: Pieces fall automatically at configurable speed
- [ ] **CORE-04**: Player can move piece left/right with arrow keys
- [ ] **CORE-05**: Player can soft drop (accelerate fall) with down arrow
- [ ] **CORE-06**: Player can hard drop (instant fall) with spacebar
- [ ] **CORE-07**: Player can rotate piece clockwise with up arrow
- [ ] **CORE-08**: Rotation uses wall kicks when near edges
- [ ] **CORE-09**: Pieces lock when they land on surface/other pieces
- [ ] **CORE-10**: Completed rows clear and award points
- [ ] **CORE-11**: Game ends when pieces stack to top

### Enhanced Mechanics

- [ ] **EHNC-01**: Ghost piece shows landing position
- [ ] **EHNC-02**: Hold piece allows swapping current piece (once per drop)
- [ ] **EHNC-03**: Next piece preview displays upcoming piece
- [ ] **EHNC-04**: Player can pause/resume game with P key

### Unique Mechanics

- [ ] **UNIQ-01**: Game alternates 10s play / 10s freeze cycles
- [ ] **UNIQ-02**: Freeze state has clear visual indicator and countdown
- [ ] **UNIQ-03**: Board grows (wider and taller) every 30s
- [ ] **UNIQ-04**: Board growth is visually smooth, pieces stay in place
- [ ] **UNIQ-05**: Board has maximum size limit

### Scoring & Progression

- [ ] **SCOR-01**: Score display updates in real-time
- [ ] **SCOR-02**: Clearing a row awards configurable points (default 10)
- [ ] **SCOR-03**: Level display shows current level
- [ ] **SCOR-04**: 100 points advances to next level
- [ ] **SCOR-05**: Level up triggers visual theme change

### Theming

- [ ] **THEM-01**: 3 pre-built themes available (Classic, Neon, Retro)
- [ ] **THEM-02**: Themes define piece colors and shapes
- [ ] **THEM-03**: Theme changes apply instantly without restart
- [ ] **THEM-04**: Level up cycles through themes

### Admin Panel

- [ ] **ADMN-01**: Admin panel runs in separate browser tab
- [ ] **ADMN-02**: Admin can select active theme from dropdown
- [ ] **ADMN-03**: Admin can adjust fall speed via slider
- [ ] **ADMN-04**: Admin can adjust points per row via input
- [ ] **ADMN-05**: Admin can adjust board growth interval via slider
- [ ] **ADMN-06**: Admin sees live game stats (score, level)
- [ ] **ADMN-07**: All admin changes sync to game in real-time

### Technical

- [ ] **TECH-01**: Game uses HTML5 Canvas for rendering
- [ ] **TECH-02**: Game runs at 60fps using requestAnimationFrame
- [ ] **TECH-03**: Admin/game sync via BroadcastChannel API
- [ ] **TECH-04**: No external dependencies (vanilla JS)
- [ ] **TECH-05**: Canvas renders crisp on high-DPI displays

## v2 Requirements

### Enhanced Features

- **V2-01**: T-spin detection with bonus points
- **V2-02**: Combo multipliers for chain clears
- **V2-03**: Session statistics (total pieces, lines, time)
- **V2-04**: Additional themes (5+ total)

### Polish

- **V2-05**: Sound effects and music
- **V2-06**: Keyboard remapping

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multiplayer | Same-browser only by design |
| Save/load game | Session-only gameplay |
| Leaderboards | Not requested |
| User accounts | Not requested |
| Mobile touch controls | Desktop browser target |
| Custom theme editor | Pre-built themes only for v1 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORE-01 | Phase 1 | Complete |
| CORE-02 | Phase 1 | Complete |
| CORE-03 | Phase 1 | Complete |
| CORE-04 | Phase 1 | Complete |
| CORE-05 | Phase 1 | Complete |
| CORE-06 | Phase 1 | Complete |
| CORE-07 | Phase 1 | Complete |
| CORE-08 | Phase 1 | Complete |
| CORE-09 | Phase 1 | Complete |
| CORE-10 | Phase 1 | Complete |
| CORE-11 | Phase 1 | Complete |
| EHNC-01 | Phase 2 | Complete |
| EHNC-02 | Phase 2 | Complete |
| EHNC-03 | Phase 2 | Complete |
| EHNC-04 | Phase 2 | Complete |
| SCOR-01 | Phase 2 | Complete |
| SCOR-02 | Phase 2 | Complete |
| SCOR-03 | Phase 2 | Complete |
| SCOR-04 | Phase 2 | Complete |
| SCOR-05 | Phase 2 | Complete |
| THEM-01 | Phase 3 | Pending |
| THEM-02 | Phase 3 | Pending |
| THEM-03 | Phase 3 | Pending |
| THEM-04 | Phase 3 | Pending |
| ADMN-01 | Phase 3 | Pending |
| ADMN-02 | Phase 3 | Pending |
| ADMN-03 | Phase 3 | Pending |
| ADMN-04 | Phase 3 | Pending |
| ADMN-05 | Phase 3 | Pending |
| ADMN-06 | Phase 3 | Pending |
| ADMN-07 | Phase 3 | Pending |
| TECH-01 | Phase 1 | Complete |
| TECH-02 | Phase 1 | Complete |
| TECH-03 | Phase 3 | Pending |
| TECH-04 | Phase 1 | Complete |
| TECH-05 | Phase 1 | Complete |
| UNIQ-01 | Phase 4 | Pending |
| UNIQ-02 | Phase 4 | Pending |
| UNIQ-03 | Phase 4 | Pending |
| UNIQ-04 | Phase 4 | Pending |
| UNIQ-05 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 37 total
- Mapped to phases: 37
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-02*
*Last updated: 2026-02-02 after Phase 2 complete*

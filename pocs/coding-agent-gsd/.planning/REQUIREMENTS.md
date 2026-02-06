# Requirements: Tetris Twist

**Defined:** 2026-02-02
**Core Value:** Real-time admin control loop — admin tweaks, player experiences instantly

## v1 Requirements (SHIPPED)

### Core Mechanics

- [x] **CORE-01**: Game displays 10-column, 20-row grid on canvas
- [x] **CORE-02**: 7 standard tetrominoes spawn (I, O, T, S, Z, J, L)
- [x] **CORE-03**: Pieces fall automatically at configurable speed
- [x] **CORE-04**: Player can move piece left/right with arrow keys
- [x] **CORE-05**: Player can soft drop (accelerate fall) with down arrow
- [x] **CORE-06**: Player can hard drop (instant fall) with spacebar
- [x] **CORE-07**: Player can rotate piece clockwise with up arrow
- [x] **CORE-08**: Rotation uses wall kicks when near edges
- [x] **CORE-09**: Pieces lock when they land on surface/other pieces
- [x] **CORE-10**: Completed rows clear and award points
- [x] **CORE-11**: Game ends when pieces stack to top

### Enhanced Mechanics

- [x] **EHNC-01**: Ghost piece shows landing position
- [x] **EHNC-02**: Hold piece allows swapping current piece (once per drop)
- [x] **EHNC-03**: Next piece preview displays upcoming piece
- [x] **EHNC-04**: Player can pause/resume game with P key

### Unique Mechanics

- [x] **UNIQ-01**: Game alternates 10s play / 10s freeze cycles
- [x] **UNIQ-02**: Freeze state has clear visual indicator and countdown
- [x] **UNIQ-03**: Board grows (wider and taller) every 30s
- [x] **UNIQ-04**: Board growth is visually smooth, pieces stay in place
- [x] **UNIQ-05**: Board has maximum size limit

### Scoring & Progression

- [x] **SCOR-01**: Score display updates in real-time
- [x] **SCOR-02**: Clearing a row awards configurable points (default 10)
- [x] **SCOR-03**: Level display shows current level
- [x] **SCOR-04**: 100 points advances to next level
- [x] **SCOR-05**: Level up triggers visual theme change

### Theming

- [x] **THEM-01**: 3 pre-built themes available (Classic, Neon, Retro)
- [x] **THEM-02**: Themes define piece colors and shapes
- [x] **THEM-03**: Theme changes apply instantly without restart
- [x] **THEM-04**: Level up cycles through themes

### Admin Panel

- [x] **ADMN-01**: Admin panel runs in separate browser tab
- [x] **ADMN-02**: Admin can select active theme from dropdown
- [x] **ADMN-03**: Admin can adjust fall speed via slider
- [x] **ADMN-04**: Admin can adjust points per row via input
- [x] **ADMN-05**: Admin can adjust board growth interval via slider
- [x] **ADMN-06**: Admin sees live game stats (score, level)
- [x] **ADMN-07**: All admin changes sync to game in real-time

### Technical

- [x] **TECH-01**: Game uses HTML5 Canvas for rendering
- [x] **TECH-02**: Game runs at 60fps using requestAnimationFrame
- [x] **TECH-03**: Admin/game sync via BroadcastChannel API
- [x] **TECH-04**: No external dependencies (vanilla JS)
- [x] **TECH-05**: Canvas renders crisp on high-DPI displays

## v2 Requirements

### T-Spin Detection

- [x] **TSPN-01**: Game detects T-spin when T-piece locks after rotation with 3+ corners occupied
- [x] **TSPN-02**: Mini T-spin detected when only 1 front corner occupied
- [x] **TSPN-03**: Full T-spin detected when 2 front corners occupied
- [x] **TSPN-04**: T-spin awards bonus points (Guideline: mini 100/200/400, full 400/800/1200/1600 × level)
- [x] **TSPN-05**: Visual indicator displays T-spin type on detection

### Combo System

- [x] **COMB-01**: Combo counter tracks consecutive line clears
- [x] **COMB-02**: Combo resets when piece locks without clearing lines
- [x] **COMB-03**: Combo awards bonus points (50 × combo × level)
- [x] **COMB-04**: Visual combo counter displays during active combo
- [x] **COMB-05**: Back-to-Back bonus (1.5x) for consecutive Tetris/T-spin clears

### Session Statistics

- [x] **STAT-01**: Track basic stats (score, lines, level, time, pieces placed)
- [x] **STAT-02**: Track advanced stats (PPS, APM, efficiency, tetris rate)
- [x] **STAT-03**: Display real-time stats in sidebar
- [x] **STAT-04**: Session summary screen shows all stats on game over

### Additional Themes

- [x] **THEM-05**: Add Minimalist theme (clean, simple colors)
- [x] **THEM-06**: Add High Contrast theme (accessibility-focused)
- [x] **THEM-07**: Admin theme selector shows all 5+ themes

### Audio

- [x] **AUDIO-01**: Sound effect plays on piece land
- [x] **AUDIO-02**: Sound effect plays on line clear
- [x] **AUDIO-03**: Sound effect plays on Tetris (4-line clear)
- [x] **AUDIO-04**: Sound effect plays on game over
- [x] **AUDIO-05**: Mute toggle in admin panel persists to localStorage

### Keyboard Remapping

- [x] **KEYS-01**: All game controls are remappable
- [x] **KEYS-02**: Visual settings UI for key binding
- [x] **KEYS-03**: Key bindings persist to localStorage
- [x] **KEYS-04**: Conflict detection prevents duplicate bindings
- [x] **KEYS-05**: Default bindings restore option

## v3 Requirements

### Combo Pitch Scaling

- [ ] **PITCH-01**: Line clear sound pitch increases proportionally with combo counter
- [ ] **PITCH-02**: Pitch scaling uses smooth parameter ramping to prevent clicks
- [ ] **PITCH-03**: Pitch capped at 10x combo to prevent painful frequencies
- [ ] **PITCH-04**: Different pitch patterns for single/double/triple/Tetris clears
- [ ] **PITCH-05**: Pitch resets to base when combo breaks

### Personal Best Tracking

- [ ] **BEST-01**: Track high score, lines, level across sessions in localStorage
- [ ] **BEST-02**: Display personal best comparison at game over screen
- [ ] **BEST-03**: Show "New Record" visual notification when beating personal best
- [ ] **BEST-04**: Track timestamp of when personal best was achieved
- [ ] **BEST-05**: Clear/reset personal bests option in admin panel

### Key Binding Export/Import

- [ ] **EXPORT-01**: Export current key bindings to tetris_keybindings.json file
- [ ] **EXPORT-02**: Import key bindings from JSON file with validation
- [ ] **EXPORT-03**: Show error messages for invalid import files
- [ ] **EXPORT-04**: Export includes metadata (date, game version)
- [ ] **EXPORT-05**: Export/Import buttons in admin panel Controls section

### Background Music

- [ ] **MUSIC-01**: Procedural background music via continuous OscillatorNode
- [ ] **MUSIC-02**: Seamless looping without restart gaps
- [ ] **MUSIC-03**: Separate mute toggle from sound effects
- [ ] **MUSIC-04**: Music pauses when game pauses, stops on game over
- [ ] **MUSIC-05**: Music gain lower than effects to not overpower gameplay

## v4 Requirements (Deferred)

### Future Polish

- **V4-01**: Adaptive music changing based on game state/level
- **V4-02**: Multiple personal best categories (best score, best lines, best PPS)
- **V4-03**: Preset key binding slots (save/load multiple profiles)
- **V4-04**: Music tempo synced to piece fall speed

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multiplayer | Same-browser only by design |
| Save/load game | Session-only gameplay |
| Leaderboards | Not requested |
| User accounts | Not requested |
| Mobile touch controls | Desktop browser target |
| Custom theme editor | Pre-built themes only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORE-01 | Phase 1 | ✓ Complete |
| CORE-02 | Phase 1 | ✓ Complete |
| CORE-03 | Phase 1 | ✓ Complete |
| CORE-04 | Phase 1 | ✓ Complete |
| CORE-05 | Phase 1 | ✓ Complete |
| CORE-06 | Phase 1 | ✓ Complete |
| CORE-07 | Phase 1 | ✓ Complete |
| CORE-08 | Phase 1 | ✓ Complete |
| CORE-09 | Phase 1 | ✓ Complete |
| CORE-10 | Phase 1 | ✓ Complete |
| CORE-11 | Phase 1 | ✓ Complete |
| EHNC-01 | Phase 2 | ✓ Complete |
| EHNC-02 | Phase 2 | ✓ Complete |
| EHNC-03 | Phase 2 | ✓ Complete |
| EHNC-04 | Phase 2 | ✓ Complete |
| SCOR-01 | Phase 2 | ✓ Complete |
| SCOR-02 | Phase 2 | ✓ Complete |
| SCOR-03 | Phase 2 | ✓ Complete |
| SCOR-04 | Phase 2 | ✓ Complete |
| SCOR-05 | Phase 2 | ✓ Complete |
| THEM-01 | Phase 3 | ✓ Complete |
| THEM-02 | Phase 3 | ✓ Complete |
| THEM-03 | Phase 3 | ✓ Complete |
| THEM-04 | Phase 3 | ✓ Complete |
| ADMN-01 | Phase 3 | ✓ Complete |
| ADMN-02 | Phase 3 | ✓ Complete |
| ADMN-03 | Phase 3 | ✓ Complete |
| ADMN-04 | Phase 3 | ✓ Complete |
| ADMN-05 | Phase 3 | ✓ Complete |
| ADMN-06 | Phase 3 | ✓ Complete |
| ADMN-07 | Phase 3 | ✓ Complete |
| TECH-01 | Phase 1 | ✓ Complete |
| TECH-02 | Phase 1 | ✓ Complete |
| TECH-03 | Phase 3 | ✓ Complete |
| TECH-04 | Phase 1 | ✓ Complete |
| TECH-05 | Phase 1 | ✓ Complete |
| UNIQ-01 | Phase 4 | ✓ Complete |
| UNIQ-02 | Phase 4 | ✓ Complete |
| UNIQ-03 | Phase 4 | ✓ Complete |
| UNIQ-04 | Phase 4 | ✓ Complete |
| UNIQ-05 | Phase 4 | ✓ Complete |
| THEM-05 | Phase 5 | ✓ Complete |
| THEM-06 | Phase 5 | ✓ Complete |
| THEM-07 | Phase 5 | ✓ Complete |
| STAT-01 | Phase 6 | ✓ Complete |
| STAT-02 | Phase 6 | ✓ Complete |
| STAT-03 | Phase 6 | ✓ Complete |
| STAT-04 | Phase 6 | ✓ Complete |
| COMB-01 | Phase 7 | ✓ Complete |
| COMB-02 | Phase 7 | ✓ Complete |
| COMB-03 | Phase 7 | ✓ Complete |
| COMB-04 | Phase 7 | ✓ Complete |
| COMB-05 | Phase 7 | ✓ Complete |
| TSPN-01 | Phase 8 | ✓ Complete |
| TSPN-02 | Phase 8 | ✓ Complete |
| TSPN-03 | Phase 8 | ✓ Complete |
| TSPN-04 | Phase 8 | ✓ Complete |
| TSPN-05 | Phase 8 | ✓ Complete |
| AUDIO-01 | Phase 9 | ✓ Complete |
| AUDIO-02 | Phase 9 | ✓ Complete |
| AUDIO-03 | Phase 9 | ✓ Complete |
| AUDIO-04 | Phase 9 | ✓ Complete |
| AUDIO-05 | Phase 9 | ✓ Complete |
| KEYS-01 | Phase 10 | ✓ Complete |
| KEYS-02 | Phase 10 | ✓ Complete |
| KEYS-03 | Phase 10 | ✓ Complete |
| KEYS-04 | Phase 10 | ✓ Complete |
| KEYS-05 | Phase 10 | ✓ Complete |
| PITCH-01 | Phase 11 | Pending |
| PITCH-02 | Phase 11 | Pending |
| PITCH-03 | Phase 11 | Pending |
| PITCH-04 | Phase 11 | Pending |
| PITCH-05 | Phase 11 | Pending |
| BEST-01 | Phase 12 | Pending |
| BEST-02 | Phase 12 | Pending |
| BEST-03 | Phase 12 | Pending |
| BEST-04 | Phase 12 | Pending |
| BEST-05 | Phase 12 | Pending |
| EXPORT-01 | Phase 13 | Pending |
| EXPORT-02 | Phase 13 | Pending |
| EXPORT-03 | Phase 13 | Pending |
| EXPORT-04 | Phase 13 | Pending |
| EXPORT-05 | Phase 13 | Pending |
| MUSIC-01 | Phase 14 | Pending |
| MUSIC-02 | Phase 14 | Pending |
| MUSIC-03 | Phase 14 | Pending |
| MUSIC-04 | Phase 14 | Pending |
| MUSIC-05 | Phase 14 | Pending |

**Coverage:**
- v1 requirements: 37 total (all complete)
- v2 requirements: 27 total (all complete)
- v3 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0

---
*Requirements defined: 2026-02-02*
*Last updated: 2026-02-06 with v3.0 phase mappings*

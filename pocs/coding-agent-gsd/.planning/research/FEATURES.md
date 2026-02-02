# Features Research: Browser Tetris Game

## Table Stakes (Must Have)

These are expected in any Tetris game. Missing them feels broken.

| Feature | Complexity | Notes |
|---------|------------|-------|
| 7 standard tetrominoes (I, O, T, S, Z, J, L) | Low | Core game identity |
| Piece rotation (clockwise) | Medium | Wall kicks add complexity |
| Piece movement (left, right, down) | Low | Basic controls |
| Hard drop (instant down) | Low | Expected shortcut |
| Line clearing | Medium | Core scoring mechanic |
| Game over detection | Low | Board overflow |
| Score display | Low | Feedback loop |
| Level display | Low | Progress indicator |
| Next piece preview | Low | Planning aid |
| Pause functionality | Low | Basic UX |

## Differentiators (This Project's Unique Features)

| Feature | Complexity | Notes |
|---------|------------|-------|
| Play/freeze cycle (10s/10s) | Medium | Timer management, visual feedback |
| Growing board | High | Dynamic canvas resize, piece repositioning |
| Real-time admin controls | Medium | BroadcastChannel sync |
| Live theme switching | Medium | CSS variables, canvas redraw |
| Pre-built themes | Low | Asset preparation |
| Admin panel UI | Medium | Separate page, config controls |

## Nice-to-Have (v2)

| Feature | Complexity | Notes |
|---------|------------|-------|
| Ghost piece (landing preview) | Low | Shows where piece will land |
| Hold piece | Medium | Swap current piece |
| T-spin detection | High | Advanced scoring |
| Combo multipliers | Medium | Chain line clears |
| Statistics (pieces placed, lines cleared) | Low | Session data |

## Anti-Features (Deliberately Excluded)

| Feature | Why Exclude |
|---------|-------------|
| Sound/music | Not requested, adds complexity |
| Multiplayer/networking | Same browser only |
| Save/load game | Session-only by design |
| Leaderboards | Not requested |
| User accounts | Not requested |
| Mobile touch controls | Web browser desktop target |

## Feature Dependencies

```
Core Tetris Mechanics
  └── Piece rendering
  └── Board state
  └── Collision detection
  └── Line clearing
        └── Scoring
              └── Level progression
                    └── Theme changes

Admin System
  └── BroadcastChannel sync
        └── Config state
              └── Theme switching
              └── Speed changes
              └── Scoring changes
              └── Board growth rate

Unique Mechanics
  └── Timer system
        └── Freeze cycle
        └── Board growth trigger

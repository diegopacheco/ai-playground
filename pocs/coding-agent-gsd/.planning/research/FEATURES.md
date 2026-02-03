# Features Research: Tetris Twist v2.0

**Domain:** Enhanced Tetris scoring mechanics and polish features
**Researched:** 2026-02-03
**Overall confidence:** MEDIUM to HIGH

This research covers T-spin detection, combo systems, session statistics, themes, sound effects, and keyboard remapping for Tetris Twist v2.0.

## T-Spin Detection

### Detection Rules (3-Corner T Rule)

T-spin detection uses the "3-corner T rule" employed by SRS guideline games:

**Basic Detection:**
- Last maneuver of T tetromino must be a rotation
- 3 of 4 squares diagonally adjacent to T's center must be occupied
- If no rotation occurred last, no T-spin regardless of position

**Mini vs Full T-Spin:**
- **Full T-Spin:** 2+ front corners occupied (front = adjacent to pointing mino)
- **Mini T-Spin:** Only 1 front corner occupied, 2 back corners filled
- **Exception:** Last rotation with 1x2 block kick (SRS offset) = always full T-spin

### Point Values (Guideline Standard)

| Action | Base Points | Formula |
|--------|------------|---------|
| T-Spin (no clear) | 400 | 400 × (level + 1) |
| T-Spin Mini Single | 200 | 200 × (level + 1) |
| T-Spin Single | 800 | 800 × (level + 1) |
| T-Spin Mini Double | 400 | 400 × (level + 1) |
| T-Spin Double | 1200 | 1200 × (level + 1) |
| T-Spin Triple | 1600 | 1600 × (level + 1) |

All T-spins that clear lines qualify for Back-to-Back bonus (1.5x multiplier).
T-spins with no clears don't break B2B chain but don't trigger B2B either.

**Confidence:** HIGH - Based on official Tetris Guideline standards

## Combo System

### Combo Mechanics

**Combo Counter Behavior:**
- Starts at -1 at game start
- Increments by 1 each time a piece clears at least one line
- Resets to -1 immediately when a piece clears no lines
- Continues across multiple consecutive line clears

**Guideline Scoring Formula:**
```
Combo points = 50 × combo_count × level
```

**Example Combo Sequence:**
1. Place piece, clear 1 line → combo = 0 → 50 × 0 × level = 0 points
2. Place piece, clear 2 lines → combo = 1 → 50 × 1 × level = 50 × level
3. Place piece, clear 1 line → combo = 2 → 50 × 2 × level = 100 × level
4. Place piece, no clear → combo = -1 → chain broken

### Combo Display

**Visual Feedback:**
- Display "COMBO x[number]" when combo >= 1
- Flash or animate on combo increase
- Show combo bonus points separately from line clear points
- Reset animation when broken

**Confidence:** HIGH - Standard guideline implementation

## Back-to-Back Bonus

### B2B Trigger Conditions

**"Difficult" clears that trigger/maintain B2B:**
- Tetris (4-line clear)
- Any T-Spin that clears lines (including Mini)
- All-Clear (Perfect Clear) in some implementations

**"Easy" clears that break B2B:**
- Single (1-line clear with non-T piece)
- Double (2-line clear with non-T piece)
- Triple (3-line clear with non-T piece)

**B2B Multiplier:** 1.5x applied to base score

**Example:**
- Tetris: 800 × (level + 1)
- B2B Tetris: 1200 × (level + 1) [+50%]
- T-Spin Double: 1200 × (level + 1)
- B2B T-Spin Double: 1800 × (level + 1) [+50%]

**Confidence:** HIGH - Verified standard

## Session Statistics

### Core Metrics (Table Stakes)

| Metric | Definition | Display Format |
|--------|-----------|----------------|
| Score | Total accumulated points | Numeric (e.g., 125,430) |
| Lines | Total lines cleared | Numeric |
| Level | Current level | Numeric |
| Time | Session duration | MM:SS or HH:MM:SS |
| Pieces | Total pieces placed | Numeric |

### Advanced Metrics (Differentiators)

| Metric | Definition | Formula | Display |
|--------|-----------|---------|---------|
| PPS | Pieces Per Second | pieces / time_seconds | 2.45 PPS |
| APM | Actions Per Minute | total_inputs / time_minutes | 180 APM |
| Tetris Rate | % of lines from Tetrises | (tetris_lines / total_lines) × 100 | 75% |
| Efficiency | Lines per piece | total_lines / total_pieces | 1.85 |
| Max Combo | Longest combo chain | Track highest combo | Max: 12 |
| T-Spins | Total T-spins executed | Count all T-spins | 15 T-Spins |

### Real-Time vs Session End

**During Gameplay (HUD):**
- Score (always visible)
- Lines (always visible)
- Level (always visible)
- Time (always visible)
- Current combo (when active)
- Current B2B status (when active)

**Session Summary (End Screen):**
- All core metrics
- All advanced metrics
- Personal bests comparison
- Grade/rank if applicable

### Performance Benchmarks

**PPS Context:**
- Beginner: 0.5-1.5 PPS
- Intermediate: 1.5-3.0 PPS
- Advanced: 3.0-5.0 PPS
- Expert: 5.0-7.0 PPS
- World-class: 10+ PPS

**Confidence:** MEDIUM - PPS benchmarks from competitive play, exact thresholds vary by source

## Additional Themes

### Theme Structure Components

**Visual Elements:**
- Background (static or animated)
- Tetromino colors (7 colors for IJLOSTZ)
- Grid color and style
- Ghost piece rendering
- UI panel styling
- Particle effects for line clears
- Border and frame design

### Existing Themes (v1.0)
1. Classic - Traditional look
2. Neon - Bright cyberpunk aesthetic
3. Retro - Nostalgic palette

### Recommended New Themes (v2.0)

**Theme 4: Minimalist/Modern**
- Clean white/gray background
- Subtle tetromino colors with borders
- Thin grid lines
- Emphasis on clarity and readability
- Target audience: Professional/work-friendly

**Theme 5: High Contrast**
- Strong black background
- Vibrant, saturated colors
- Bold outlines
- Maximum visibility
- Accessibility-focused

**Theme 6: Nature/Organic**
- Earth tones palette
- Wood or stone textures
- Green/brown/cream colors
- Calming aesthetic
- Casual player appeal

### Color Palette Standards

**Traditional Tetromino Colors:**
- I: Cyan (#00FFFF)
- O: Yellow (#FFFF00)
- T: Purple (#800080)
- S: Green (#00FF00)
- Z: Red (#FF0000)
- J: Blue (#0000FF)
- L: Orange (#FF7F00)

**Theme Differentiation:**
- Maintain color distinctiveness (avoid confusion)
- Ensure ghost piece visibility against background
- Test contrast ratios for accessibility
- Keep UI elements readable in all themes

**Complexity:** LOW - Mostly CSS/color value changes

**Confidence:** HIGH - Standard practice

## Sound Effects

### Essential Sound Events (Table Stakes)

| Event | Purpose | Characteristics |
|-------|---------|-----------------|
| Piece land | Feedback for placement | Short percussive stinger |
| Line clear | Reward for clearing | Pitched stinger, varies by count |
| Tetris | Special reward | Longer, more dramatic |
| T-Spin | Special achievement | Unique recognizable sound |
| Level up | Milestone celebration | Ascending tone or fanfare |
| Game over | Session end | Descending tone or impact |
| Piece rotate | Movement feedback | Subtle click/tick |
| Piece move | Optional movement feedback | Very subtle, can be omitted |

### Advanced Sound Events (Differentiators)

| Event | Purpose | Implementation |
|-------|---------|----------------|
| Combo increment | Reward combo building | Ascending pitch per combo level |
| B2B trigger | Difficult clear reward | Layered effect on existing clear sound |
| Hold piece | Swap feedback | Soft whoosh or swap sound |
| Ghost piece toggle | UI feedback | Quick UI confirmation |
| Theme change | UI feedback | Transition sound |

### Sound Design Principles

**Tetris Effect Approach:**
- Different pitches for directional movement
- Percussive stingers sync with placement
- Line clear sounds vary by line count
- Music integration (optional advanced feature)

**Frequency Considerations:**
- High-frequency events (rotate, move) must be non-intrusive
- Reward sounds (clear, Tetris) should feel satisfying
- Volume balance critical for playability
- All sounds must be toggleable (mute option)

**Anti-Pattern to Avoid:**
- Don't make every input sound annoying at high PPS
- Avoid repetitive sounds that become grating
- Don't make sounds mandatory (always allow mute)

**Complexity:** MEDIUM - Requires sound asset creation/sourcing and event integration

**Confidence:** MEDIUM - Based on Tetris Effect analysis and general practice

## Keyboard Remapping

### Standard Default Controls

**Movement:**
- Left Arrow: Move left
- Right Arrow: Move right
- Down Arrow: Soft drop
- Space: Hard drop

**Rotation:**
- Up Arrow: Rotate clockwise
- Z/Ctrl: Rotate counter-clockwise

**Actions:**
- Shift/C: Hold piece
- Escape/P: Pause

### Common Alternative Schemes

**WASD-style:**
- A/D: Move left/right
- S: Soft drop
- W: Hard drop
- Q/E: Rotate

**Gaming Optimal:**
- SDFG: Move/drop controls
- W/E: Rotate
- Spacebar: Hard drop
- Shift: Hold

### Remapping Requirements (Table Stakes)

**Core Features:**
- All gameplay keys remappable
- In-game settings menu for remapping
- Click-to-rebind interface
- Conflict detection (prevent duplicate bindings)
- Reset to defaults option
- Visual display of current bindings

**Technical Implementation:**
- Store keybindings in localStorage
- Support keyCode and key event properties
- Handle modifier keys (Shift, Ctrl, Alt)
- Prevent system key conflicts (F5, Ctrl+W, etc.)
- Test mode to verify new bindings work

### UX Best Practices

**Remapping Flow:**
1. Click on action to rebind
2. Press new key
3. Show confirmation or conflict warning
4. Save immediately to localStorage
5. Allow cancel/undo

**Display:**
- Show key name, not keyCode
- Use visual key representations
- Group by category (movement, rotation, actions)
- Indicate conflicts clearly
- Show defaults alongside custom bindings

### Modern Expectations (2026)

**Essential:**
- Full keyboard remapping has become standard
- Settings should persist across sessions
- No need for external configuration files
- In-game interface expected over config files

**Nice-to-Have:**
- Preset profiles (Default, WASD, Gaming, etc.)
- Import/export keybindings
- Multiple control schemes switchable mid-session
- Controller support (out of scope for keyboard focus)

**Complexity:** MEDIUM - Requires settings UI, conflict detection, localStorage

**Confidence:** HIGH - Standard feature in modern games

## Feature Categories

### Table Stakes (Must-Have for v2.0)

| Feature | Rationale | Complexity |
|---------|-----------|-----------|
| T-Spin Detection (mini + full) | Expected in any modern Tetris with SRS | MEDIUM |
| T-Spin Scoring | Core reward mechanic for skilled play | LOW |
| Combo Counter Display | Visual feedback for consecutive clears | LOW |
| Combo Scoring | Standard expectation in Tetris games | LOW |
| Back-to-Back Bonus | Standard competitive mechanic | LOW |
| Basic Session Stats | Score, lines, level, time always expected | LOW |
| 2-3 New Themes | Adds variety, low cost to implement | LOW |
| Essential Sound Effects | Land, clear, Tetris, game over minimum | MEDIUM |
| Basic Keyboard Remapping | Players expect control customization | MEDIUM |

### Differentiators (Competitive Advantages)

| Feature | Value Proposition | Complexity |
|---------|------------------|-----------|
| Advanced Stats (PPS, APM, Efficiency) | Appeals to competitive players | LOW |
| Max Combo Tracking | Achievement-oriented feedback | LOW |
| T-Spin Counter | Skill-tracking for advanced players | LOW |
| Personal Best Comparison | Session-to-session progression | MEDIUM |
| Combo Sound Pitch Scaling | Enhanced audio feedback (Tetris Effect style) | MEDIUM |
| Theme Hot-Swapping | QoL feature, mid-game theme change | LOW |
| Visual Remapping Interface | Professional polish over text configs | MEDIUM |
| Session Summary Screen | Detailed post-game analysis | LOW |

### Anti-Features (Deliberately Avoid)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Auto-play or AI assistance | Defeats core gameplay loop | Provide practice modes or tutorials |
| Pay-to-win mechanics | Browser game should be free | Keep all features accessible |
| Complex achievement system | Scope creep for v2.0 | Simple stat tracking sufficient |
| Social features (leaderboards) | Requires backend infrastructure | Local personal bests only |
| Microtransactions for themes | Inappropriate for project scope | All themes unlocked |
| Mandatory sound | Accessibility issue | Always allow full mute |
| Locked keyboard layouts | Frustrates players | Full remapping support |
| Animation-heavy line clears | Can slow down fast play | Keep effects snappy |

## Feature Dependencies

### Dependency Chain

```
Core Game Loop (v1.0)
    ├─> T-Spin Detection (NEW)
    │   └─> T-Spin Scoring (NEW)
    │       └─> Back-to-Back Tracking (NEW)
    │           └─> B2B Visual Indicator (NEW)
    │
    ├─> Combo System (NEW)
    │   ├─> Combo Counter (NEW)
    │   ├─> Combo Scoring (NEW)
    │   └─> Combo Display (NEW)
    │
    ├─> Session Statistics (NEW)
    │   ├─> Basic Stats (Score, Lines, Level, Time) [v1.0]
    │   ├─> Advanced Stats (PPS, APM, Efficiency) (NEW)
    │   └─> Session Summary Screen (NEW)
    │
    ├─> Themes (EXTEND)
    │   ├─> 3 Existing Themes [v1.0]
    │   └─> 3 New Themes (NEW)
    │
    ├─> Sound Effects (NEW)
    │   ├─> Basic Sounds (land, clear, game over) (NEW)
    │   ├─> Advanced Sounds (combo, B2B, T-spin) (NEW)
    │   └─> Mute Toggle (NEW)
    │
    └─> Keyboard Remapping (NEW)
        ├─> Settings UI (NEW)
        ├─> Key Binding Storage (NEW)
        └─> Conflict Detection (NEW)
```

### Implementation Order Recommendation

**Phase 1: Scoring Mechanics (Core Value)**
1. T-Spin detection logic
2. T-Spin scoring integration
3. Combo counter logic
4. Combo scoring
5. Back-to-Back tracking
6. Visual indicators for T-Spin/Combo/B2B

**Phase 2: Feedback & Polish**
1. Basic sound effects (land, clear, game over)
2. Advanced sound effects (T-spin, combo, B2B)
3. Sound toggle controls
4. New themes (3 additional)
5. Theme switcher UI

**Phase 3: Configuration & Stats**
1. Keyboard remapping UI
2. Key binding storage
3. Conflict detection
4. Advanced session stats (PPS, APM, etc.)
5. Session summary screen
6. Personal best tracking

## MVP Recommendation

### Must-Have for v2.0 Launch

**Scoring Mechanics (Core):**
- T-Spin detection (full + mini)
- T-Spin scoring with proper multipliers
- Combo system with scoring
- Back-to-Back bonus tracking
- Visual indicators (on-screen text for "T-SPIN!" "COMBO x3" "B2B")

**Feedback:**
- 5-6 essential sound effects minimum
- Mute toggle
- 2 new themes minimum (1 differentiator acceptable)

**Configuration:**
- Basic keyboard remapping
- Settings persistence

**Statistics:**
- Time tracking (if not in v1.0)
- Session summary with basic stats

### Can Defer to v2.1+

**Advanced Polish:**
- Pitch-scaled combo sounds
- Additional theme variations
- Animated line clear effects

**Advanced Stats:**
- PPS, APM calculations
- Personal best comparisons
- Historical session tracking

**QoL Features:**
- Mid-game theme switching
- Preset control schemes
- Advanced remapping (import/export)

## Implementation Complexity Summary

| Feature Category | Complexity | Estimated Effort | Risk |
|-----------------|-----------|------------------|------|
| T-Spin Detection | MEDIUM | 4-6 hours | MEDIUM - Corner detection logic tricky |
| T-Spin Scoring | LOW | 1-2 hours | LOW - Formula-based |
| Combo System | LOW | 2-3 hours | LOW - Counter logic simple |
| B2B Tracking | LOW | 1-2 hours | LOW - State machine |
| Session Stats Basic | LOW | 2-3 hours | LOW - Already have most |
| Session Stats Advanced | LOW | 3-4 hours | LOW - Calculation overhead minimal |
| New Themes | LOW | 1-2 hours each | LOW - CSS changes |
| Sound Effects | MEDIUM | 4-8 hours | MEDIUM - Asset creation + integration |
| Keyboard Remapping | MEDIUM | 6-8 hours | MEDIUM - UI + conflict detection |

**Total Estimated Effort:** 25-40 hours for full v2.0 scope

## Sources

**T-Spin Detection & Scoring:**
- [T-Spin - TetrisWiki](https://tetris.wiki/T-Spin)
- [T-Spin Guide - Hard Drop Tetris Wiki](https://harddrop.com/wiki/T-Spin_Guide)
- [Scoring - TetrisWiki](https://tetris.wiki/Scoring)
- [T-Spin Guide | TETRIS-FAQ](https://winternebs.github.io/TETRIS-FAQ/tspin/)

**Combo System:**
- [Combo - TetrisWiki](https://tetris.wiki/Combo)
- [Combo - Hard Drop Tetris Wiki](https://harddrop.com/wiki/Combo)
- [Scoring - TetrisWiki](https://tetris.wiki/Scoring)

**Session Statistics:**
- [Statistics Portal - Liquipedia Tetris Wiki](https://liquipedia.net/tetris/Portal:Statistics)
- [TETR.IO - TetrisWiki](https://tetris.wiki/TETR.IO)
- [What Is PPS in Tetris? - Playbite](https://www.playbite.com/q/what-is-pps-tetris)
- [GitHub - ejona86/taus: Tetris - Actually Useful Statistics](https://github.com/ejona86/taus)

**Themes:**
- [Tetris Game Color Scheme - SchemeColor.com](https://www.schemecolor.com/tetris-game-color-scheme.php)
- [Tetris Basic Color Scheme - ColorsWall](https://colorswall.com/palette/90259)
- [Tetris Color Codes - Brand Palettes](https://brandpalettes.com/tetris-color-codes/)

**Sound Effects:**
- [Game audio analysis - Tetris Effect](https://www.gamedeveloper.com/audio/game-audio-analysis---tetris-effect)
- [Tetris: Adding polish with music, sound effects - Katy's Code](https://katyscode.wordpress.com/2013/03/15/tetris-adding-polish-with-music-sound-effects-backgrounds-game-options-an-intro-sequence-and-other-tweaks/)

**Keyboard Remapping:**
- [TETR.IO Controls - 9meters](https://9meters.com/entertainment/games/tetr-io-controls-essential-keyboard-shortcuts)
- [Game interface - Hard Drop Tetris Wiki](https://harddrop.com/wiki/Game_interface)

**Back-to-Back Bonus:**
- [Back-to-Back - Hard Drop Tetris Wiki](https://harddrop.com/wiki/Back-to-Back)
- [Scoring in Tetris Mobile Help Center](https://playstudios.helpshift.com/hc/en/16-tetris-mobile/faq/2437-scoring-in-tetris/)

**Implementation Pitfalls:**
- [Common Beginner Mistakes - Galactoid's Tetris Guides](https://galactoidtetris.wordpress.com/2020/09/30/common-beginner-mistakes/)
- [Types of Mechanical Mistakes in Tetris - Medium](https://medium.com/@lucas_bomfim/types-of-mechanical-mistakes-in-tetris-42c59264ed82)

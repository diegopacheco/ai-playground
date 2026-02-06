# Architecture Research: Tetris Twist v3.0

**Researched:** 2026-02-06
**Focus:** Audio polish and persistence features

## Current Architecture Summary

Tetris Twist v2.0 uses a vanilla JS architecture with clear module boundaries:

| Module | Responsibility | State Management |
|--------|---------------|------------------|
| main.js | Game loop, state machine (GameState enum), event dispatch, combo tracking | Game state variables |
| audio.js | Web Audio API (OscillatorNode), playSound function, mute state | audioContext, muted flag |
| stats.js | Session statistics (score, lines, PPS, APM, maxCombo, b2bCount, tSpinCount) | stats object |
| input.js | Key binding management, DAS/ARR input handling | keymap object, localStorage |
| admin.js | Admin panel logic, BroadcastChannel sync | currentKeymap |
| sync.js | BroadcastChannel message passing | channel singleton |
| render.js | Canvas drawing, UI display | Visual only, no state |
| board.js | Board state, collision detection | board array |

**Key patterns:**
- Global state per module (var-scoped)
- BroadcastChannel for cross-tab sync
- localStorage for persistence (audio_muted, tetris_keybindings)
- Event-driven updates via BroadcastChannel messages

## Integration Analysis

### Combo Pitch Scaling

**Existing module:** audio.js
**Integration point:** main.js combo variable

**Changes needed:**
1. Modify audio.js:
   - Add function: playLineClearSound(comboValue)
   - Scale frequency based on combo: baseFreq + (comboValue * pitchStep)
   - Keep existing playSound function as internal implementation

2. Modify main.js:
   - Update line clear logic in lockPieceToBoard function
   - Pass combo value to audio call: playLineClearSound(combo)

**Data flow:**
```
main.js (combo tracking)
  → lockPieceToBoard detects line clear
  → playLineClearSound(combo)
  → audio.js scales pitch based on combo value
  → OscillatorNode plays scaled frequency
```

**Implementation complexity:** LOW
- Existing combo variable already tracked in main.js
- Existing audio infrastructure supports frequency parameters
- Single function signature change

**Dependencies:** None - combo system already implemented in v2.0

---

### Background Music

**Existing module:** audio.js
**Integration point:** main.js GameState enum

**Changes needed:**
1. Modify audio.js:
   - Add musicContext state variable
   - Add function: startBackgroundMusic()
   - Add function: stopBackgroundMusic()
   - Add function: pauseBackgroundMusic()
   - Add function: resumeBackgroundMusic()
   - Implement looping tone sequence using OscillatorNode chain
   - Respect existing muted flag

2. Modify main.js:
   - Call startBackgroundMusic() in resetGame()
   - Call pauseBackgroundMusic() when GameState.PAUSED
   - Call resumeBackgroundMusic() when GameState.PLAYING
   - Call stopBackgroundMusic() when GameState.GAME_OVER

3. Modify admin.html + admin.js:
   - Add mute checkbox affects both SFX and music
   - OR add separate music toggle (preferred for UX)

**Data flow:**
```
main.js (GameState changes)
  → update() function state transitions
  → audio.js music control functions
  → OscillatorNode loop management
  → Respects muted flag from localStorage
```

**Implementation complexity:** MEDIUM
- New audio feature (looping music vs one-shot SFX)
- State management: must track playing/paused/stopped
- Must handle AudioContext suspend/resume correctly
- Interaction with existing mute system

**Design decision required:**
- Single mute toggle for all audio OR separate music/SFX toggles?
- Recommendation: Separate toggles (localStorage keys: audio_muted, music_muted)
- Rationale: Players often want SFX but no music or vice versa

---

### Personal Best Tracking

**Existing module:** stats.js
**Integration point:** main.js game over logic

**Changes needed:**
1. Modify stats.js:
   - Add function: loadPersonalBest() returns { score, lines, level, date }
   - Add function: savePersonalBest(score, lines, level)
   - Add function: checkNewBest() compares current to stored
   - Use localStorage key: tetris_personal_best
   - JSON structure: { score: 0, lines: 0, level: 0, date: ISO8601 }

2. Modify main.js:
   - Call checkNewBest() when GameState.GAME_OVER
   - Optional: Set flag for render.js to show "NEW BEST!" indicator

3. Modify render.js:
   - Extend drawSessionSummary() to show personal best comparison
   - Display: "Personal Best: 1234 (2026-02-05)"
   - Display: "NEW BEST!" if current session beats stored

**Data flow:**
```
main.js (game over)
  → stats.checkNewBest()
  → Compare stats.score to localStorage.tetris_personal_best
  → If higher: savePersonalBest() updates localStorage
  → render.js drawSessionSummary() reads best for display
```

**Implementation complexity:** LOW
- Existing stats module already tracks all needed values
- localStorage pattern already established (audio_muted, keymap)
- No sync needed (personal best is per-browser)

**Storage structure:**
```json
{
  "score": 1234,
  "lines": 45,
  "level": 13,
  "date": "2026-02-06T14:23:45.678Z"
}
```

---

### Key Binding Export/Import

**Existing module:** input.js + admin.js
**Integration point:** admin.html UI

**Changes needed:**
1. Modify input.js:
   - Add function: exportKeymap() returns JSON string
   - Add function: importKeymap(jsonString) validates and applies
   - Add function: validateKeymapStructure(obj) checks required actions
   - Reuse existing saveKeymap() for persistence

2. Modify admin.js:
   - Add export button handler: trigger download of JSON file
   - Add import button handler: file input → read → importKeymap()
   - Use existing currentKeymap variable
   - Use existing saveAndSync() for BroadcastChannel updates

3. Modify admin.html:
   - Add "Export Key Bindings" button in Controls section
   - Add "Import Key Bindings" button + hidden file input
   - Add feedback message area for import success/failure

**Data flow:**
```
Export:
  admin.html button click
  → admin.js reads currentKeymap
  → input.js exportKeymap() serializes to JSON
  → Browser downloads file: tetris_keybindings.json

Import:
  admin.html file input
  → admin.js reads file contents
  → input.js importKeymap() validates structure
  → If valid: apply to currentKeymap + saveAndSync()
  → BroadcastChannel notifies game tab
  → game tab input.js updates keymap
```

**Implementation complexity:** LOW
- Existing keymap structure already JSON-serializable
- Validation logic straightforward (check action keys exist)
- File download/upload is standard browser API
- BroadcastChannel sync already implemented

**JSON export format:**
```json
{
  "left": ["ArrowLeft"],
  "right": ["ArrowRight"],
  "down": ["ArrowDown"],
  "rotate": ["ArrowUp"],
  "hardDrop": ["Space"],
  "hold": ["KeyC"],
  "pause": ["KeyP"]
}
```

**Validation rules:**
- Must have all 7 actions (left, right, down, rotate, hardDrop, hold, pause)
- Each action must be array of strings
- Each string must be valid KeyboardEvent.code value
- No conflicts (same key in multiple actions)

---

## New Modules Required

**NONE** - All features extend existing modules:
- audio.js handles both combo pitch and background music
- stats.js handles personal best tracking
- input.js + admin.js handle key binding export/import

This maintains architectural consistency with v1.0 and v2.0 patterns.

---

## Build Order Recommendation

### Phase 1: Combo Pitch Scaling
**Rationale:** Extends existing audio.js, leverages existing combo tracking, zero dependencies

**Tasks:**
1. Modify playLineClearSound(comboValue) signature
2. Implement frequency scaling formula
3. Update main.js line clear calls
4. Test combo chains (1x → 5x should increase pitch noticeably)

**Risk:** LOW - Modifies one function signature, existing combo system stable

---

### Phase 2: Personal Best Tracking
**Rationale:** Extends existing stats.js, independent of audio changes, simple localStorage

**Tasks:**
1. Add loadPersonalBest() + savePersonalBest() to stats.js
2. Add checkNewBest() logic
3. Update main.js game over logic
4. Update render.js drawSessionSummary() display
5. Test: play game, check localStorage, verify best saved

**Risk:** LOW - Purely additive, no changes to existing stat tracking

---

### Phase 3: Key Binding Export/Import
**Rationale:** Extends existing input system, UI-only feature, no gameplay dependencies

**Tasks:**
1. Add exportKeymap() + importKeymap() + validation to input.js
2. Add export/import buttons to admin.html
3. Add export handler (JSON download) to admin.js
4. Add import handler (file upload + validation) to admin.js
5. Test: export default, modify, import, verify sync

**Risk:** LOW - File I/O is browser standard, existing keymap structure already JSON-clean

---

### Phase 4: Background Music
**Rationale:** Most complex, requires new audio state management, test last to avoid breaking SFX

**Tasks:**
1. Design looping tone sequence (melody + rhythm)
2. Add music state management to audio.js
3. Add start/stop/pause/resume functions
4. Integrate with GameState transitions in main.js
5. Add music mute toggle to admin.html + admin.js
6. Test state transitions: play → pause → resume → game over
7. Test mute interaction: SFX muted but music playing, etc.

**Risk:** MEDIUM - New audio state, complex interaction with existing mute system, GameState coordination

---

## Technical Decisions

### Decision 1: Separate Music Mute Toggle
**Options:**
- A: Single mute toggle affects both SFX and music
- B: Separate toggles for SFX and music

**Recommendation:** B (Separate toggles)

**Rationale:**
- Players often want different settings (SFX on, music off is common)
- localStorage supports multiple keys easily
- Admin UI has space for both checkboxes
- More flexible UX

**Implementation:**
- localStorage keys: audio_muted, music_muted
- audio.js checks both flags independently
- admin.html adds second checkbox in Audio section

---

### Decision 2: Personal Best Scope
**Options:**
- A: Track only highest score
- B: Track score, lines, level, date

**Recommendation:** B (Track multiple metrics)

**Rationale:**
- Stats module already tracks all values
- Date provides context for achievements
- Negligible localStorage cost
- Enables richer UI ("Beat your 1234 from 3 days ago!")

**Implementation:**
- JSON object with 4 fields
- Compare on score only (primary metric)
- Display all fields in game over summary

---

### Decision 3: Combo Pitch Scaling Formula
**Options:**
- A: Linear: baseFreq + (combo * 50Hz)
- B: Exponential: baseFreq * (1.1 ^ combo)
- C: Stepped: baseFreq + (Math.floor(combo/2) * 100Hz)

**Recommendation:** A (Linear with cap)

**Formula:** 440Hz + (Math.min(combo, 10) * 40Hz)
- Combo 1: 440Hz (A4)
- Combo 5: 640Hz (E5)
- Combo 10+: 840Hz (capped for comfort)

**Rationale:**
- Linear feels intuitive (higher combo = higher pitch)
- Cap prevents painful high frequencies
- 40Hz steps are perceptible but not jarring
- Matches existing simple audio design

---

### Decision 4: Key Binding Export Filename
**Options:**
- A: Generic: "keybindings.json"
- B: Timestamped: "keybindings_20260206_142345.json"
- C: Branded: "tetris_keybindings.json"

**Recommendation:** C (Branded, no timestamp)

**Rationale:**
- User may export multiple configs (not need timestamp)
- Branding helps identify file purpose
- Simple name easier to share/rename
- Matches localStorage key convention (tetris_keybindings)

---

## Integration Points Summary

| Feature | Primary Module | Secondary Modules | Integration Complexity |
|---------|---------------|-------------------|----------------------|
| Combo pitch scaling | audio.js | main.js | LOW - 1 function signature change |
| Background music | audio.js | main.js, admin.js, admin.html | MEDIUM - new state management |
| Personal best tracking | stats.js | main.js, render.js | LOW - additive only |
| Key binding export/import | input.js | admin.js, admin.html | LOW - file I/O + validation |

---

## Confidence Assessment

**Overall: HIGH**

All features integrate cleanly with existing architecture:
- No new modules required (extends existing)
- Patterns match v1.0/v2.0 precedent (localStorage, BroadcastChannel)
- Dependencies minimal (combo pitch uses existing combo tracking)
- Build order clear (independent features can parallelize)

**Risks identified:**
1. Background music state management most complex
2. Music/mute interaction needs careful testing
3. Key binding validation must be robust (prevent broken configs)

**Mitigation:**
- Build music feature last (after simpler features validated)
- Separate music toggle reduces mute interaction complexity
- Validation function with clear error messages
- Test plan should cover all state transitions

---

## Architectural Consistency Notes

v3.0 maintains architectural patterns from v1.0 and v2.0:

**Preserved patterns:**
- Global state per module (var-scoped)
- localStorage for persistence (mute, keymap, now personal best)
- BroadcastChannel for cross-tab sync
- Vanilla JS with zero dependencies
- Function-based modules (no classes)
- Feature flags through localStorage

**No breaking changes:**
- All existing functions keep signatures (except playLineClearSound)
- Additive only (no removals)
- Backward compatible (missing localStorage keys use defaults)

This continuity reduces integration risk and maintains codebase simplicity.

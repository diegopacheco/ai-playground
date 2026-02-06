# Stack Research: Tetris Twist v3.0

**Researched:** 2026-02-06
**Focus:** Audio polish and persistence features
**Confidence:** HIGH

## Executive Summary

v3.0 requires ZERO new libraries or frameworks. All features achievable with existing Web Audio API, localStorage, and File API patterns already validated in v2.0. Combo pitch scaling uses existing OscillatorNode frequency adjustment. Background music requires AudioBufferSourceNode with loop configuration. Personal bests extend existing localStorage patterns. Key binding export/import uses native Blob and File APIs.

## Existing Stack (Validated in v2.0)

| Component | Technology | Current Usage |
|-----------|------------|---------------|
| Audio synthesis | Web Audio API (OscillatorNode) | Sound effects with frequency/duration parameters |
| Audio context | AudioContext | Initialized on user interaction, suspended state management |
| State persistence | localStorage | Audio mute state, key bindings (JSON serialized) |
| Cross-window sync | BroadcastChannel | Admin/game synchronization for settings |
| Game rendering | Canvas 2D API | Board, pieces, UI elements |
| Input handling | KeyboardEvent + DAS timing | Configurable key bindings with repeat logic |

**Validated capabilities:**
- `playSound(frequency, duration)` creates short sound effects
- `localStorage.getItem/setItem` with JSON.stringify/parse for objects
- Keymap stored as: `{ action: [keyCodes] }`
- Combo tracking in game loop with `updateComboStats(combo)`

## New Capabilities Needed

### Combo Pitch Scaling

**Requirement:** Sound effect pitch increases with combo count.

**Implementation approach:**
- Modify existing `playSound(frequency, duration)` to accept combo parameter
- Scale frequency based on combo: `baseFrequency * (1 + (combo * scaleFactor))`
- No new APIs needed, OscillatorNode.frequency.value is directly writable

**Code pattern:**
```javascript
function playLineClearSound(combo) {
    const baseFreq = 440;
    const scaleFactor = 0.1;
    const scaledFreq = baseFreq * (1 + (combo * scaleFactor));
    playSound(scaledFreq, 100);
}
```

**Technical details:**
- OscillatorNode frequency is an AudioParam (a-rate)
- Direct value assignment: `oscillator.frequency.value = newFrequency`
- Alternative (precise timing): `oscillator.frequency.setValueAtTime(freq, time)`
- For real-time combo changes, direct value assignment is sufficient
- Frequency range: keep between 110 Hz - 2000 Hz for playability

**Source:** [MDN OscillatorNode.frequency](https://developer.mozilla.org/en-US/docs/Web/API/OscillatorNode/frequency)

**Confidence:** HIGH (existing OscillatorNode already validated)

### Background Music

**Requirement:** Looping audio during gameplay.

**Implementation approach:**
- Use AudioBufferSourceNode instead of OscillatorNode
- Set `loop = true` for continuous playback
- Load audio buffer from file or generate procedurally

**Decision: Procedural vs Static Audio**

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| Static file (MP3/WAV) | Rich sound, familiar melodies | File size (100KB-2MB), loading time, licensing | Skip for MVP |
| Procedural (OscillatorNode) | Zero file size, no licensing, already validated | Limited musical complexity | Use for MVP |
| No background music | Zero complexity | Missing feature | Acceptable fallback |

**Recommended: Procedural generation using existing OscillatorNode**

**Code pattern:**
```javascript
let musicOscillator = null;
let musicGain = null;

function startBackgroundMusic() {
    const ctx = getAudioContext();
    musicOscillator = ctx.createOscillator();
    musicGain = ctx.createGain();

    musicOscillator.connect(musicGain);
    musicGain.connect(ctx.destination);

    musicOscillator.frequency.value = 330;
    musicOscillator.type = 'sine';
    musicGain.gain.value = 0.05;

    musicOscillator.start();
}

function stopBackgroundMusic() {
    if (musicOscillator) {
        musicOscillator.stop();
        musicOscillator = null;
    }
}
```

**Alternative (static audio):**
If static files chosen later, use AudioBufferSourceNode:
```javascript
const response = await fetch("music.wav");
const buffer = await ctx.decodeAudioData(await response.arrayBuffer());

const source = ctx.createBufferSource();
source.buffer = buffer;
source.loop = true;
source.loopStart = 0;
source.loopEnd = buffer.duration;
source.connect(ctx.destination);
source.start();
```

**Important:** AudioBufferSourceNode is one-shot. To restart, create new node.

**Sources:**
- [MDN AudioBufferSourceNode.loop](https://developer.mozilla.org/en-US/docs/Web/API/AudioBufferSourceNode/loop)
- [MDN Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)

**Confidence:** HIGH (OscillatorNode approach), MEDIUM (static file approach needs testing)

### Personal Best Persistence

**Requirement:** Track and persist personal records across sessions.

**Implementation approach:**
- Extend localStorage patterns already validated for keymap
- Store stats object with schema version for migration
- Load on game initialization, update on game over

**JSON structure:**
```javascript
{
    "version": 1,
    "personalBests": {
        "highScore": 0,
        "maxLines": 0,
        "maxCombo": 0,
        "bestPPS": 0,
        "bestAPM": 0,
        "totalGamesPlayed": 0
    },
    "lastUpdated": "2026-02-06T12:00:00.000Z"
}
```

**Storage pattern:**
```javascript
function loadPersonalBests() {
    try {
        const stored = localStorage.getItem('tetris_personal_bests');
        if (stored) {
            const data = JSON.parse(stored);
            return migrateSchema(data);
        }
        return getDefaultBests();
    } catch (e) {
        return getDefaultBests();
    }
}

function savePersonalBests(bests) {
    const data = {
        version: 1,
        personalBests: bests,
        lastUpdated: new Date().toISOString()
    };
    localStorage.setItem('tetris_personal_bests', JSON.stringify(data));
}

function migrateSchema(data) {
    if (!data.version) {
        data = { version: 1, personalBests: data };
    }
    return data;
}
```

**Migration strategy:**
- Include version field for schema evolution
- Implement migration function to handle old formats
- Default to empty state on parse errors (no corruption)

**Storage limits:**
- localStorage quota: 5-10MB per origin (browser dependent)
- Personal bests JSON: approximately 200 bytes
- No quota concerns for this feature

**Sources:**
- [localStorage Guide - Meticulous](https://www.meticulous.ai/blog/localstorage-complete-guide)
- [localStorage Best Practices 2026](https://copyprogramming.com/howto/javascript-how-ot-keep-local-storage-on-refresh)

**Confidence:** HIGH (localStorage already validated, patterns established)

### Key Binding Export/Import

**Requirement:** Download key bindings as JSON, upload JSON to restore bindings.

**Implementation approach:**
- Export: Blob + URL.createObjectURL + anchor click
- Import: File input + FileReader or Blob.text()
- Validate imported JSON structure before applying

**Export pattern:**
```javascript
function exportKeyBindings() {
    const keymap = getKeymap();
    const exportData = {
        version: 1,
        keybindings: keymap,
        exportedAt: new Date().toISOString()
    };

    const jsonString = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "tetris-keybindings.json";
    link.click();

    URL.revokeObjectURL(url);
}
```

**Import pattern:**
```javascript
function setupImportButton() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json';

    fileInput.addEventListener('change', async () => {
        const file = fileInput.files[0];
        if (!file) return;

        try {
            const jsonString = await file.text();
            const data = JSON.parse(jsonString);

            if (validateKeybindings(data)) {
                applyKeyBindings(data.keybindings);
                saveKeymap();
            }
        } catch (e) {
            alert('Invalid keybindings file');
        }
    });

    importButton.addEventListener('click', () => fileInput.click());
}

function validateKeybindings(data) {
    return data.version &&
           data.keybindings &&
           typeof data.keybindings === 'object';
}
```

**Security considerations:**
- Validate JSON structure before applying
- Sanitize key codes (check against known key code format)
- Handle malformed JSON gracefully
- No XSS risk (only applying key codes to keymap, not rendering HTML)

**Browser compatibility:**
- Blob API: Universal support
- URL.createObjectURL: Universal support
- File API: Universal support
- Blob.text(): Modern browsers (fallback: FileReader.readAsText)

**Sources:**
- [MDN File API](https://developer.mozilla.org/en-US/docs/Web/API/File_API)
- [json-porter library reference](https://github.com/markgab/json-porter)

**Confidence:** HIGH (standard File API patterns, no new libraries)

## Integration Points

### Audio Module (audio.js)
**Existing:** `playSound(frequency, duration)` for sound effects
**New integration:**
- Add combo parameter to sound effect functions
- Add `startBackgroundMusic()` and `stopBackgroundMusic()`
- Maintain existing mute state handling

### Stats Module (stats.js)
**Existing:** Session stats tracking, combo tracking
**New integration:**
- Add `loadPersonalBests()` on initialization
- Add `updatePersonalBests(sessionStats)` on game over
- Add comparison functions (isNewRecord)

### Input Module (input.js)
**Existing:** `loadKeymap()`, `saveKeymap()`, `getKeymap()`
**New integration:**
- Add `exportKeyBindings()` function
- Add `importKeyBindings(file)` function
- Add validation function for imported bindings

### Admin Module (admin.js)
**Existing:** Theme controls, audio mute toggle, key binding UI
**New integration:**
- Add export/import buttons to Controls section
- Wire up click handlers for export/import
- Display success/error feedback

## Stack Additions

**None required.**

All features achievable with native Web APIs already in use:
- Web Audio API (validated)
- localStorage (validated)
- File API (standard browser API, no library needed)
- JSON (native JavaScript)

## Anti-Recommendations

### Do NOT Add These:

**Tone.js or Howler.js (audio libraries)**
- Why: Adds 50-100KB for capabilities already available in Web Audio API
- Existing OscillatorNode handles all current requirements
- Only consider if moving to complex music generation (not in scope)

**localForage or Dexie.js (storage libraries)**
- Why: Adds complexity for simple key-value persistence
- Personal bests data is <1KB, no need for IndexedDB
- localStorage quota (5-10MB) is sufficient

**FileSaver.js (file download library)**
- Why: Native Blob + URL.createObjectURL works in all modern browsers
- Adds dependency for single-function use case
- 3KB library to save 5 lines of code

**Web Audio API polyfills**
- Why: Target browsers (2026) have universal Web Audio support
- OscillatorNode and AudioBufferSourceNode are baseline features
- No legacy browser support needed

**JSON schema validation libraries (ajv, joi)**
- Why: Simple structure validation is 5 lines of code
- Schema is trivial (version + object + ISO timestamp)
- Adds build complexity for minimal benefit

## Version Verification

All APIs verified as current for 2026:

| API | Status | Browser Support |
|-----|--------|-----------------|
| Web Audio API | Stable | Universal (2011+ baseline) |
| OscillatorNode | Stable | Universal |
| AudioBufferSourceNode | Stable | Universal |
| localStorage | Stable | Universal |
| File API | Stable | Universal |
| Blob.text() | Stable | Modern browsers (Chrome 76+, Firefox 69+) |
| URL.createObjectURL | Stable | Universal |

**Fallback for Blob.text():**
```javascript
const text = await file.text();

const reader = new FileReader();
reader.onload = () => processJSON(reader.result);
reader.readAsText(file);
```

## Implementation Order Recommendation

**Phase 1: Audio polish**
1. Combo pitch scaling (simplest, extends existing)
2. Background music (procedural OscillatorNode approach)

**Phase 2: Persistence**
3. Personal best tracking (localStorage pattern established)
4. Key binding export/import (requires UI changes)

**Rationale:** Audio features are independent and extend existing patterns. Persistence features build on each other (both use localStorage and JSON).

## Confidence Assessment

| Area | Confidence | Reasoning |
|------|------------|-----------|
| Combo pitch scaling | HIGH | OscillatorNode.frequency.value validated in v2.0 |
| Procedural music | HIGH | Same OscillatorNode pattern, continuous instead of transient |
| Static audio files | MEDIUM | AudioBufferSourceNode not yet validated, requires fetch + decode |
| Personal best persistence | HIGH | localStorage + JSON validated in keymap.js |
| Export/import | HIGH | File API is standard, pattern is straightforward |
| Overall stack | HIGH | Zero new dependencies, all APIs validated or standard |

## Risk Assessment

**Low risk:**
- All features use validated Web APIs
- No external dependencies to maintain
- No build process changes
- Backwards compatible (features are additive)

**Potential issues:**
- Background music may be annoying (needs volume control, toggle)
- File import validation needs testing with malformed JSON
- Frequency scaling needs tuning for good audio feel

**Mitigation:**
- Add background music mute toggle (separate from SFX mute)
- Comprehensive validation with try-catch and user feedback
- Playtesting with various combo ranges (1x to 20x+)

## Sources

This research verified current API status and patterns from:
- [MDN Web Audio API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [MDN OscillatorNode.frequency](https://developer.mozilla.org/en-US/docs/Web/API/OscillatorNode/frequency)
- [MDN AudioBufferSourceNode.loop](https://developer.mozilla.org/en-US/docs/Web/API/AudioBufferSourceNode/loop)
- [MDN File API](https://developer.mozilla.org/en-US/docs/Web/API/File_API)
- [localStorage Complete Guide - Meticulous](https://www.meticulous.ai/blog/localstorage-complete-guide)
- [localStorage Best Practices 2026](https://copyprogramming.com/howto/javascript-how-ot-keep-local-storage-on-refresh)
- [json-porter GitHub](https://github.com/markgab/json-porter)

# Phase 10: Keyboard Remapping - Research

**Researched:** 2026-02-05
**Domain:** JavaScript Keyboard Events / Input Mapping
**Confidence:** HIGH

## Summary

Keyboard remapping for games requires a configurable keymap architecture that decouples physical key detection from game actions. The current `input.js` hardcodes key checks directly in `getInput()`, which must be refactored to use a keymap object lookup pattern. The standard approach uses `event.code` (physical key position) rather than `event.key` (character value) to ensure consistent behavior across keyboard layouts (QWERTY, AZERTY, Dvorak).

The UI pattern for key capture involves a focused element that listens for the next keypress, validates against conflicts, and updates the binding. Conflict detection is straightforward: maintain a reverse mapping from key codes to actions, check before applying new bindings. localStorage persistence uses JSON serialization of the keymap object with a separate constant holding default values for restoration.

The existing BroadcastChannel infrastructure can synchronize key binding changes between the game and admin panel. The localStorage pattern already established for audio mute preferences applies directly to key bindings.

**Primary recommendation:** Refactor `getInput()` to use a `keymap` object mapping action names to key codes, store bindings in localStorage as JSON, add a key binding UI section to admin.html with conflict detection, and maintain a `DEFAULT_KEYMAP` constant for restoration.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| KeyboardEvent.code | Native | Physical key identification | Layout-independent, consistent across platforms |
| localStorage API | Native | Persist key bindings | Already in use for audio/theme preferences |
| BroadcastChannel API | Native | Cross-tab sync | Already in use per existing architecture |
| JSON | Native | Serialize keymap object | Standard approach for complex data in localStorage |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Object.assign / Spread | Native ES6 | Reset to defaults | Copy DEFAULT_KEYMAP to active keymap |
| Array.find | Native ES5+ | Conflict detection | Check if key already bound to another action |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| event.code | event.key | Character-based, breaks on non-QWERTY layouts |
| localStorage | IndexedDB | Overkill for simple key-value storage |
| Manual conflict check | Map/Set data structure | Adds complexity without benefit for 7 actions |

**Installation:**
None required - all APIs are native to browsers.

## Architecture Patterns

### Recommended Project Structure
```
js/
├── input.js          # Add keymap object, loadKeymap(), saveKeymap()
├── keybindings.js    # NEW: Key binding UI logic for admin panel
```

### Pattern 1: Configurable Keymap Object
**What:** Replace hardcoded key checks with object lookup
**When to use:** Always - foundation for remapping

**Current (hardcoded):**
```javascript
if (keys['ArrowLeft']) {
    // handle left
}
if (keys['ArrowRight']) {
    // handle right
}
```

**Refactored (configurable):**
```javascript
const DEFAULT_KEYMAP = {
    left: ['ArrowLeft'],
    right: ['ArrowRight'],
    down: ['ArrowDown'],
    rotate: ['ArrowUp'],
    hardDrop: ['Space'],
    hold: ['KeyC', 'ShiftLeft', 'ShiftRight'],
    pause: ['KeyP']
};

let keymap = { ...DEFAULT_KEYMAP };

function isActionPressed(action) {
    const boundKeys = keymap[action];
    return boundKeys.some(function(code) {
        return keys[code];
    });
}

function getInput() {
    const input = {
        left: false,
        right: false,
        down: false,
        rotate: false,
        hardDrop: false,
        hold: false,
        pause: false
    };

    if (isActionPressed('left')) {
        const keyCode = keymap.left.find(function(code) { return keys[code]; });
        const timer = keyTimers[keyCode];
        // DAS logic unchanged
    }
    // ... rest of actions
    return input;
}
```

### Pattern 2: Key Capture for Remapping
**What:** Modal or inline element captures next keypress for binding
**When to use:** Key binding UI in admin panel

```javascript
let capturingFor = null;

function startCapture(action) {
    capturingFor = action;
    document.getElementById('capture-' + action).textContent = 'Press a key...';
    document.addEventListener('keydown', handleCapture, { once: true });
}

function handleCapture(event) {
    event.preventDefault();
    event.stopPropagation();

    if (!capturingFor) return;

    const code = event.code;
    const conflict = checkConflict(code, capturingFor);

    if (conflict) {
        showConflictWarning(conflict, code);
        return;
    }

    setKeyBinding(capturingFor, code);
    document.getElementById('capture-' + capturingFor).textContent = getKeyDisplayName(code);
    capturingFor = null;
}
```

### Pattern 3: Conflict Detection with Reverse Lookup
**What:** Before binding, check if key is already used by another action
**When to use:** Every binding attempt

```javascript
function checkConflict(keyCode, excludeAction) {
    for (var action in keymap) {
        if (action === excludeAction) continue;
        if (keymap[action].includes(keyCode)) {
            return action;
        }
    }
    return null;
}

function showConflictWarning(conflictingAction, keyCode) {
    alert(getKeyDisplayName(keyCode) + ' is already bound to ' + conflictingAction);
}
```

### Pattern 4: localStorage Persistence with Defaults
**What:** Store bindings as JSON, restore defaults from constant
**When to use:** Save/load key bindings

```javascript
const STORAGE_KEY = 'tetris_keybindings';

function loadKeymap() {
    var stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
        try {
            keymap = JSON.parse(stored);
        } catch (e) {
            keymap = { ...DEFAULT_KEYMAP };
        }
    } else {
        keymap = { ...DEFAULT_KEYMAP };
    }
}

function saveKeymap() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(keymap));
    channel.postMessage({ type: 'KEYMAP_CHANGE', payload: { keymap: keymap } });
}

function restoreDefaults() {
    keymap = {};
    for (var action in DEFAULT_KEYMAP) {
        keymap[action] = DEFAULT_KEYMAP[action].slice();
    }
    saveKeymap();
    updateUI();
}
```

### Pattern 5: Human-Readable Key Names
**What:** Convert event.code values to display-friendly names
**When to use:** UI display of current bindings

```javascript
const KEY_DISPLAY_NAMES = {
    'ArrowLeft': 'Left Arrow',
    'ArrowRight': 'Right Arrow',
    'ArrowUp': 'Up Arrow',
    'ArrowDown': 'Down Arrow',
    'Space': 'Space',
    'ShiftLeft': 'Left Shift',
    'ShiftRight': 'Right Shift',
    'ControlLeft': 'Left Ctrl',
    'ControlRight': 'Right Ctrl',
    'KeyP': 'P',
    'KeyC': 'C',
    'KeyR': 'R',
    'Escape': 'Esc',
    'Enter': 'Enter',
    'Tab': 'Tab'
};

function getKeyDisplayName(code) {
    if (KEY_DISPLAY_NAMES[code]) {
        return KEY_DISPLAY_NAMES[code];
    }
    if (code.startsWith('Key')) {
        return code.charAt(3);
    }
    if (code.startsWith('Digit')) {
        return code.charAt(5);
    }
    return code;
}
```

### Anti-Patterns to Avoid

**Using event.key instead of event.code:** The `key` property varies by keyboard layout. Using `event.code` ensures the physical key position is consistent regardless of QWERTY, AZERTY, or Dvorak layouts.

**Storing bindings without validation:** Always validate loaded JSON from localStorage. Corrupted data should fall back to defaults.

**Overwriting defaults on load:** Do not initialize localStorage with defaults if it already has data. Check existence first.

**Hardcoding action names in UI:** Use the keymap object keys to generate UI, making it extensible.

**Forgetting to sync across tabs:** Use BroadcastChannel to ensure game and admin panel stay in sync.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Key name mapping | Manual switch statement | Lookup object + string parsing | Covers all keys, easier to maintain |
| Cross-tab sync | localStorage events | BroadcastChannel API | Real-time, already in use |
| Default restoration | Manual per-property copy | Spread operator or Object.assign | Clean, one-liner |
| JSON validation | Manual type checking | try/catch with fallback | Handles all malformed data |

**Key insight:** The keymap object pattern with array values (allowing multiple keys per action) provides flexibility for both single-key bindings and combo keys like hold (Shift or C).

## Common Pitfalls

### Pitfall 1: Using event.key for Game Controls
**What goes wrong:** Controls break when user switches keyboard layout (QWERTY to AZERTY).
**Why it happens:** `event.key` returns the character produced, which varies by layout.
**How to avoid:** Always use `event.code` for game controls.
**Warning signs:** Users on non-English keyboards report controls not working.

### Pitfall 2: Not Preventing Default During Capture
**What goes wrong:** Browser performs default action (scrolling, focusing) during key capture.
**Why it happens:** Arrow keys scroll, Tab changes focus, Space activates buttons.
**How to avoid:** Call `event.preventDefault()` in the capture handler.
**Warning signs:** Page scrolls or focus moves when user tries to bind arrow keys.

### Pitfall 3: Binding Escape or Tab to Game Actions
**What goes wrong:** User cannot close dialogs or navigate with keyboard.
**Why it happens:** These keys have important accessibility functions.
**How to avoid:** Consider blocklisting Escape and Tab from remapping, or warn user.
**Warning signs:** User cannot dismiss the key binding UI after binding Escape.

### Pitfall 4: localStorage JSON Parse Errors
**What goes wrong:** Game crashes on load if localStorage contains invalid JSON.
**Why it happens:** User manually edited localStorage, or browser extension corrupted it.
**How to avoid:** Wrap `JSON.parse()` in try/catch, fall back to defaults.
**Warning signs:** Console errors on page load, settings not persisting.

### Pitfall 5: Forgetting Array Copy for Defaults
**What goes wrong:** Modifying keymap also modifies DEFAULT_KEYMAP.
**Why it happens:** Arrays are reference types in JavaScript.
**How to avoid:** Use `slice()` or spread operator when copying: `keymap.left = DEFAULT_KEYMAP.left.slice()`.
**Warning signs:** Restore Defaults button stops working after first use.

### Pitfall 6: DAS Timer Not Transferred to New Key
**What goes wrong:** DAS (Delayed Auto Shift) breaks after remapping.
**Why it happens:** Timer lookup uses old key code that no longer matches.
**How to avoid:** DAS timer lookup must use the dynamically bound key, not hardcoded.
**Warning signs:** Holding direction key no longer auto-repeats after rebinding.

## Code Examples

Verified patterns for implementation:

### Complete Keymap Integration with DAS
```javascript
const DEFAULT_KEYMAP = {
    left: ['ArrowLeft'],
    right: ['ArrowRight'],
    down: ['ArrowDown'],
    rotate: ['ArrowUp'],
    hardDrop: ['Space'],
    hold: ['KeyC', 'ShiftLeft', 'ShiftRight'],
    pause: ['KeyP']
};

let keymap = {};

function loadKeymap() {
    var stored = localStorage.getItem('tetris_keybindings');
    if (stored) {
        try {
            keymap = JSON.parse(stored);
        } catch (e) {
            restoreDefaults();
        }
    } else {
        restoreDefaults();
    }
}

function restoreDefaults() {
    keymap = {};
    for (var action in DEFAULT_KEYMAP) {
        keymap[action] = DEFAULT_KEYMAP[action].slice();
    }
}

function getActiveKeyForAction(action) {
    var boundKeys = keymap[action];
    for (var i = 0; i < boundKeys.length; i++) {
        if (keys[boundKeys[i]]) {
            return boundKeys[i];
        }
    }
    return null;
}

function getInputWithDAS(action) {
    var keyCode = getActiveKeyForAction(action);
    if (!keyCode) return false;

    var timer = keyTimers[keyCode];
    if (!timer) return false;

    var now = Date.now();
    var elapsed = now - timer.pressed;

    if (elapsed < DAS_DELAY) {
        if (timer.lastRepeat === 0) {
            timer.lastRepeat = now;
            return true;
        }
        return false;
    } else {
        if (now - timer.lastRepeat >= DAS_REPEAT) {
            timer.lastRepeat = now;
            return true;
        }
        return false;
    }
}
```

### Key Binding UI Section for Admin Panel
```html
<div class="section">
    <h2>Controls</h2>
    <div class="key-bindings">
        <div class="binding-row">
            <span class="action-label">Move Left</span>
            <button id="bind-left" class="key-button">Left Arrow</button>
        </div>
        <div class="binding-row">
            <span class="action-label">Move Right</span>
            <button id="bind-right" class="key-button">Right Arrow</button>
        </div>
        <div class="binding-row">
            <span class="action-label">Soft Drop</span>
            <button id="bind-down" class="key-button">Down Arrow</button>
        </div>
        <div class="binding-row">
            <span class="action-label">Rotate</span>
            <button id="bind-rotate" class="key-button">Up Arrow</button>
        </div>
        <div class="binding-row">
            <span class="action-label">Hard Drop</span>
            <button id="bind-hardDrop" class="key-button">Space</button>
        </div>
        <div class="binding-row">
            <span class="action-label">Hold Piece</span>
            <button id="bind-hold" class="key-button">C / Shift</button>
        </div>
        <div class="binding-row">
            <span class="action-label">Pause</span>
            <button id="bind-pause" class="key-button">P</button>
        </div>
    </div>
    <button id="restore-defaults" class="restore-btn">Restore Defaults</button>
</div>
```

### Key Binding JavaScript for Admin Panel
```javascript
var channel = new BroadcastChannel('tetris-sync');
var capturingAction = null;
var currentKeymap = {};

var ACTION_LABELS = {
    left: 'Move Left',
    right: 'Move Right',
    down: 'Soft Drop',
    rotate: 'Rotate',
    hardDrop: 'Hard Drop',
    hold: 'Hold Piece',
    pause: 'Pause'
};

function initKeyBindingUI() {
    var stored = localStorage.getItem('tetris_keybindings');
    if (stored) {
        try {
            currentKeymap = JSON.parse(stored);
        } catch (e) {
            currentKeymap = getDefaultKeymap();
        }
    } else {
        currentKeymap = getDefaultKeymap();
    }
    updateAllBindingButtons();

    for (var action in ACTION_LABELS) {
        setupBindButton(action);
    }

    document.getElementById('restore-defaults').addEventListener('click', function() {
        currentKeymap = getDefaultKeymap();
        saveAndSync();
        updateAllBindingButtons();
    });
}

function setupBindButton(action) {
    var btn = document.getElementById('bind-' + action);
    btn.addEventListener('click', function() {
        startCapture(action, btn);
    });
}

function startCapture(action, button) {
    capturingAction = action;
    button.textContent = 'Press a key...';
    button.classList.add('capturing');
    document.addEventListener('keydown', handleKeyCapture);
}

function handleKeyCapture(event) {
    event.preventDefault();
    event.stopPropagation();

    if (!capturingAction) return;

    var code = event.code;
    var conflict = findConflict(code, capturingAction);

    if (conflict) {
        alert(getKeyDisplayName(code) + ' is already bound to ' + ACTION_LABELS[conflict]);
        cancelCapture();
        return;
    }

    currentKeymap[capturingAction] = [code];
    saveAndSync();

    var btn = document.getElementById('bind-' + capturingAction);
    btn.textContent = getKeyDisplayName(code);
    btn.classList.remove('capturing');

    document.removeEventListener('keydown', handleKeyCapture);
    capturingAction = null;
}

function cancelCapture() {
    if (!capturingAction) return;
    var btn = document.getElementById('bind-' + capturingAction);
    btn.textContent = getKeyDisplayName(currentKeymap[capturingAction][0]);
    btn.classList.remove('capturing');
    document.removeEventListener('keydown', handleKeyCapture);
    capturingAction = null;
}

function findConflict(code, excludeAction) {
    for (var action in currentKeymap) {
        if (action === excludeAction) continue;
        if (currentKeymap[action].includes(code)) {
            return action;
        }
    }
    return null;
}

function saveAndSync() {
    localStorage.setItem('tetris_keybindings', JSON.stringify(currentKeymap));
    channel.postMessage({ type: 'KEYMAP_CHANGE', payload: { keymap: currentKeymap } });
}

function updateAllBindingButtons() {
    for (var action in currentKeymap) {
        var btn = document.getElementById('bind-' + action);
        if (btn && currentKeymap[action].length > 0) {
            var names = currentKeymap[action].map(getKeyDisplayName);
            btn.textContent = names.join(' / ');
        }
    }
}
```

### CSS for Key Binding UI
```css
.key-bindings {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.binding-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.action-label {
    flex: 1;
}
.key-button {
    min-width: 120px;
    padding: 8px 12px;
    background: #2a2a4a;
    color: #fff;
    border: 1px solid #00ffff;
    border-radius: 4px;
    cursor: pointer;
    font-family: monospace;
}
.key-button:hover {
    background: #3a3a5a;
}
.key-button.capturing {
    background: #00ffff;
    color: #000;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
.restore-btn {
    margin-top: 15px;
    padding: 10px;
    background: #ff4444;
    color: #fff;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    width: 100%;
}
.restore-btn:hover {
    background: #ff6666;
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| event.keyCode | event.code | 2016 | keyCode deprecated, code is layout-independent |
| event.which | event.code or event.key | 2016 | which deprecated along with keyCode |
| Hardcoded keys | Configurable keymap | Best practice | Enables user customization |
| localStorage events | BroadcastChannel | 2016-2021 | Real-time sync without disk I/O |

**Deprecated/outdated:**
- **event.keyCode:** Numeric codes, deprecated. Use `event.code` or `event.key`.
- **event.which:** Same as keyCode, deprecated.
- **keypress event:** Deprecated for detecting non-character keys. Use `keydown`.

## Open Questions

Things that couldn't be fully resolved:

1. **Multi-key bindings for single action**
   - What we know: Current hold action accepts C, ShiftLeft, or ShiftRight.
   - What's unclear: Should remapping UI allow multiple keys per action, or simplify to one?
   - Recommendation: Start with single key per action for simplicity. Array structure allows future expansion.

2. **Escape key handling**
   - What we know: Escape is commonly used for menus/modals.
   - What's unclear: Should users be allowed to bind Escape to game actions?
   - Recommendation: Allow it but warn user; document that Escape may have system uses.

3. **Modifier key combinations**
   - What we know: Current system treats ShiftLeft as a standalone key.
   - What's unclear: Should users be able to bind Ctrl+X or Shift+A combinations?
   - Recommendation: Keep it simple - single keys only. Combinations add complexity without clear benefit for Tetris.

## Sources

### Primary (HIGH confidence)
- [MDN: KeyboardEvent.code](https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/code) - Physical key identification
- [MDN: Keyboard event code values](https://developer.mozilla.org/en-US/docs/Web/API/UI_Events/Keyboard_event_code_values) - Standard code values
- [javascript.info: Keyboard Events](https://javascript.info/keyboard-events) - event.code vs event.key comparison
- Existing codebase analysis: input.js, admin.js, sync.js

### Secondary (MEDIUM confidence)
- [Stephen Dodd: Keyboard Event Game Input Map](https://stephendoddtech.com/blog/game-design/keyboard-event-game-input-map) - Game keymap architecture
- [MDN: Element keydown event](https://developer.mozilla.org/en-US/docs/Web/API/Element/keydown_event) - Event handling patterns
- [Samantha Ming: Default Values](https://www.samanthaming.com/tidbits/52-3-ways-to-set-default-value/) - Default restoration patterns

### Tertiary (LOW confidence)
- [GitHub: KeyboardJS](https://github.com/RobertWHurst/KeyboardJS) - Key binding library patterns (not recommended for use due to no-dependency constraint)
- [Mousetrap](https://craig.is/killing/mice) - Keyboard shortcut patterns (reference only)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All native browser APIs, well-documented
- Architecture: HIGH - Pattern derived from existing codebase patterns and MDN docs
- Pitfalls: HIGH - Common issues documented in MDN and verified against existing code
- Code examples: HIGH - Patterns follow existing codebase conventions

**Research date:** 2026-02-05
**Valid until:** 2026-03-05 (30 days - stable APIs)

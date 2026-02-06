# Phase 12: Personal Best Tracking - Research

**Researched:** 2026-02-06
**Domain:** Browser game data persistence with localStorage
**Confidence:** HIGH

## Summary

Personal best tracking in browser-based games is a well-established pattern that leverages localStorage for persistent data storage across sessions. The standard approach involves storing high scores, statistics, and timestamps as serialized JSON objects in localStorage, comparing current session results against stored bests, and displaying comparison data with visual feedback for new records.

The implementation is straightforward in vanilla JavaScript, requiring no external dependencies. Key technical considerations include proper error handling for localStorage operations, JSON serialization/deserialization patterns, and canvas-based UI overlays for displaying comparisons and celebrations.

**Primary recommendation:** Store personal bests as a single namespaced JSON object in localStorage, implement try-catch error handling for all storage operations, use Date.now() for timestamps, and enhance the existing game over screen with side-by-side comparison displays and celebration animations for new records.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| localStorage | Native API | Persistent browser storage | Native browser API, no dependencies needed, perfect for small data like high scores |
| JSON | Native | Object serialization | Native JavaScript, standard for localStorage complex data storage |
| Date | Native | Timestamp generation | Native JavaScript, provides millisecond-precision timestamps |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| performance.now() | Native API | High-precision timing | For measuring session duration, already used in stats.js |
| requestAnimationFrame | Native API | Smooth animations | For number counting animations and celebration effects |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| localStorage | IndexedDB | IndexedDB supports 50+ GB vs localStorage's 5-10 MB, but massive overkill for simple high scores |
| Date.now() | performance.now() | performance.now() offers microsecond precision, but Date.now() is better for wall-clock timestamps |
| Native JSON | External library | No benefit for simple object serialization |

**Installation:**
No installation required - all native browser APIs.

## Architecture Patterns

### Recommended Project Structure
```
js/
├── stats.js          # Session statistics (existing)
├── bestRecords.js    # NEW: Personal best tracking module
└── render.js         # Game over screen rendering (existing, extend)
```

### Pattern 1: Namespaced localStorage Keys
**What:** Use a consistent prefix for all localStorage keys to avoid conflicts with other scripts or libraries.
**When to use:** Always, especially when third-party services might also use localStorage.
**Example:**
```javascript
const STORAGE_KEY = 'tetris_personal_bests';

function savePersonalBests(bests) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(bests));
        return true;
    } catch (e) {
        console.error('Failed to save personal bests:', e);
        return false;
    }
}

function loadPersonalBests() {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (!stored) return null;
        return JSON.parse(stored);
    } catch (e) {
        console.error('Failed to load personal bests:', e);
        return null;
    }
}
```

### Pattern 2: Single Object Storage
**What:** Store all personal best data as a single JSON object rather than separate keys.
**When to use:** When tracking multiple related metrics (score, lines, level, timestamp).
**Example:**
```javascript
const personalBests = {
    highScore: 0,
    mostLines: 0,
    highestLevel: 1,
    timestamp: Date.now()
};
```

### Pattern 3: Defensive JSON Parsing
**What:** Always wrap JSON.parse in try-catch and provide fallback values.
**When to use:** Every time you read from localStorage.
**Example:**
```javascript
function getPersonalBests() {
    try {
        const stored = localStorage.getItem('tetris_personal_bests');
        if (stored) {
            return JSON.parse(stored);
        }
    } catch (e) {
        console.error('Failed to parse personal bests:', e);
    }

    return {
        highScore: 0,
        mostLines: 0,
        highestLevel: 1,
        timestamp: null
    };
}
```

### Pattern 4: Compare and Update Pattern
**What:** Load existing bests, compare with current session, update and save if new record.
**When to use:** At game over, when final session stats are known.
**Example:**
```javascript
function checkAndUpdatePersonalBests(sessionStats) {
    const bests = loadPersonalBests() || createDefaultBests();
    const newRecords = {
        score: false,
        lines: false,
        level: false
    };

    if (sessionStats.score > bests.highScore) {
        bests.highScore = sessionStats.score;
        bests.timestamp = Date.now();
        newRecords.score = true;
    }

    if (sessionStats.lines > bests.mostLines) {
        bests.mostLines = sessionStats.lines;
        newRecords.lines = true;
    }

    if (sessionStats.level > bests.highestLevel) {
        bests.highestLevel = sessionStats.level;
        newRecords.level = true;
    }

    if (newRecords.score || newRecords.lines || newRecords.level) {
        savePersonalBests(bests);
    }

    return { bests, newRecords };
}
```

### Pattern 5: Canvas Overlay for Game Over Screen
**What:** Extend existing drawSessionSummary to include personal best comparison.
**When to use:** When rendering the game over screen.
**Example:**
```javascript
function drawSessionSummary(board, score, level) {
    const boxWidth = 280;
    const boxHeight = 410;

    const { bests, newRecords } = getComparisonData();

    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'left';
    ctx.fillText('Current:', boxX + 20, statsY);
    ctx.textAlign = 'right';
    ctx.fillStyle = newRecords.score ? '#00ff00' : '#00ffff';
    ctx.fillText(score.toString(), boxX + boxWidth - 20, statsY);

    ctx.textAlign = 'left';
    ctx.fillStyle = '#888888';
    ctx.fillText('Best:', boxX + 20, statsY + 15);
    ctx.textAlign = 'right';
    ctx.fillText(bests.highScore.toString(), boxX + boxWidth - 20, statsY + 15);
}
```

### Anti-Patterns to Avoid
- **Using clear() for selective deletion:** Never use localStorage.clear() to reset personal bests; it will delete ALL localStorage including keybindings and mute settings. Use removeItem() instead.
- **Storing individual keys separately:** Don't use separate keys like 'tetris_high_score', 'tetris_high_lines', etc. Use a single namespaced object.
- **Forgetting try-catch:** localStorage operations can throw exceptions (quota exceeded, security errors in private mode), always wrap in try-catch.
- **Assuming localStorage exists:** localStorage might be disabled or unavailable; always check before using.
- **Storing Date objects directly:** JSON.stringify converts Date objects to strings, use Date.now() timestamps instead.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Number counting animation | Custom setInterval loop | requestAnimationFrame with easing | RAF provides smoother 60fps animation, better performance |
| Timestamp formatting | Manual string concatenation | Date methods with toLocaleString or custom formatters | Handles edge cases, time zones, localization |
| Storage availability detection | Simple try-catch on first use | Initialization check with feature detection | Detects quota exceeded, private browsing, disabled storage |
| Deep object comparison | Nested if statements | Comparison by serialization or dedicated comparison | Handles nested objects, null/undefined edge cases |

**Key insight:** localStorage operations have many edge cases (quota exceeded, private browsing mode, disabled storage, malformed JSON) that require comprehensive error handling. The browser provides no external libraries specifically for high score tracking because the native APIs are sufficient when used correctly.

## Common Pitfalls

### Pitfall 1: QuotaExceededError Not Handled
**What goes wrong:** setItem throws when localStorage quota (5-10 MB) is exceeded, causing unhandled exceptions.
**Why it happens:** Developers assume unlimited storage or don't expect other scripts to consume quota.
**How to avoid:** Always wrap setItem in try-catch and handle the specific error.
**Warning signs:** Game crashes or stops saving data after extended play or when other sites have filled quota.
```javascript
try {
    localStorage.setItem(key, value);
} catch (e) {
    if (e instanceof DOMException && (
        e.code === 22 ||
        e.name === 'QuotaExceededError' ||
        e.name === 'NS_ERROR_DOM_QUOTA_REACHED'
    )) {
        console.error('localStorage quota exceeded');
    }
}
```

### Pitfall 2: Private Browsing Mode Breaks Storage
**What goes wrong:** Some browsers throw exceptions when accessing localStorage in private/incognito mode.
**Why it happens:** Private browsing disables persistent storage for privacy.
**How to avoid:** Wrap all localStorage access in try-catch, provide fallback for unavailable storage.
**Warning signs:** Game works in normal mode but breaks in incognito mode.

### Pitfall 3: Comparing Numbers as Strings
**What goes wrong:** JSON.parse might not correctly type numbers, leading to string comparison ("10" > "9" is false).
**Why it happens:** Forgetting to parse numbers after JSON deserialization.
**How to avoid:** Ensure stored values have correct types, use parseInt/parseFloat if needed.
**Warning signs:** Lower scores incorrectly identified as new records.
```javascript
const bests = JSON.parse(stored);
if (typeof bests.highScore !== 'number') {
    bests.highScore = parseInt(bests.highScore, 10) || 0;
}
```

### Pitfall 4: Using clear() Instead of removeItem()
**What goes wrong:** localStorage.clear() removes ALL keys including keybindings, mute settings, etc.
**Why it happens:** Developer wants to reset personal bests but doesn't realize clear() affects all data.
**How to avoid:** Use removeItem('tetris_personal_bests') for selective deletion.
**Warning signs:** Resetting personal bests also resets user preferences.

### Pitfall 5: Timestamp Precision Mismatch
**What goes wrong:** Mixing Date.now() (wall clock) with performance.now() (monotonic timer) causes timestamp inconsistencies.
**Why it happens:** Both return numbers in milliseconds but measure different things.
**How to avoid:** Use Date.now() for wall-clock timestamps that survive page reloads.
**Warning signs:** Timestamps don't match real-world time or become negative.

### Pitfall 6: Race Conditions with Multiple Tabs
**What goes wrong:** Multiple game tabs can overwrite each other's localStorage updates.
**Why it happens:** localStorage has no locking mechanism, last write wins.
**How to avoid:** Accept this limitation or implement storage event listeners for synchronization.
**Warning signs:** Personal bests occasionally reset or don't persist correctly.

## Code Examples

Verified patterns from research and existing codebase:

### localStorage Read Pattern (from existing js/input.js)
```javascript
try {
    var stored = localStorage.getItem('tetris_keybindings');
    if (stored) {
        keymap = JSON.parse(stored);
    } else {
        keymap = getDefaultKeymap();
    }
} catch (e) {
    keymap = getDefaultKeymap();
}
```

### localStorage Write Pattern (from existing js/input.js)
```javascript
localStorage.setItem('tetris_keybindings', JSON.stringify(keymap));
```

### Timestamp Storage Pattern
```javascript
const personalBests = {
    highScore: 0,
    mostLines: 0,
    highestLevel: 1,
    timestamp: Date.now()
};

localStorage.setItem('tetris_personal_bests', JSON.stringify(personalBests));
```

### Timestamp Display Pattern
```javascript
function formatTimestamp(timestamp) {
    if (!timestamp) return 'Never';
    const date = new Date(timestamp);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}
```

### Number Counting Animation Pattern
```javascript
function animateNumber(element, start, end, duration) {
    const startTime = performance.now();
    const range = end - start;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(start + range * easeOut);

        element.textContent = current.toString();

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}
```

### Canvas Text Flash Animation Pattern
```javascript
function drawNewRecordBadge(ctx, x, y, time) {
    const pulseSpeed = 1000;
    const opacity = 0.5 + 0.5 * Math.sin((time / pulseSpeed) * Math.PI * 2);

    ctx.globalAlpha = opacity;
    ctx.fillStyle = '#00ff00';
    ctx.font = 'bold 20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('NEW RECORD!', x, y);
    ctx.globalAlpha = 1.0;
}
```

### Side-by-Side Comparison Pattern (from existing render.js)
```javascript
function drawComparison(ctx, label, current, previous, x, y, isNewRecord) {
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'left';
    ctx.fillText(label + ':', x, y);

    ctx.textAlign = 'right';
    ctx.fillStyle = isNewRecord ? '#00ff00' : '#00ffff';
    ctx.fillText(current.toString(), x + 150, y);

    ctx.fillStyle = '#888888';
    ctx.font = '12px Arial';
    ctx.fillText('(prev: ' + previous + ')', x + 150, y + 15);
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Cookies for persistence | localStorage API | 2010s (HTML5) | Larger storage (5-10 MB vs 4 KB), simpler API |
| Separate keys per metric | Single namespaced JSON object | Ongoing best practice | Easier management, atomic updates |
| Synchronous storage assumed safe | Try-catch on all operations | 2020s (private browsing awareness) | Handles edge cases, prevents crashes |
| Manual string formatting | Date.toLocaleString() | 2015+ (better browser support) | Handles localization, time zones |
| setInterval for animations | requestAnimationFrame | 2010s (modern browsers) | 60fps smooth, better performance |

**Deprecated/outdated:**
- Using cookies for game state persistence: localStorage is now universally supported and provides more storage.
- Storing each metric as a separate localStorage key: Single object pattern is cleaner and atomic.
- Using alert() for storage errors: Console logging is standard for development, silent fallback for production.

## Open Questions

Things that couldn't be fully resolved:

1. **Canvas animation library necessity**
   - What we know: Simple pulse/flash effects can be done with native canvas APIs, existing codebase has no animation library dependencies.
   - What's unclear: Whether a library like GSAP would significantly improve celebration animations.
   - Recommendation: Start with native canvas animations (opacity pulse, scale), only add library if complex animations are needed later.

2. **Multi-tab synchronization complexity**
   - What we know: localStorage doesn't lock across tabs, storage events can detect changes from other tabs.
   - What's unclear: Whether this game needs multi-tab sync for personal bests.
   - Recommendation: Ignore multi-tab for MVP, check and update on game over only. Add storage event listener if users report issues.

3. **Storage format versioning**
   - What we know: Data structure might evolve (add new metrics).
   - What's unclear: Whether to include version number in stored object.
   - Recommendation: Include a simple version field for future migration support, even if not used immediately.

## Sources

### Primary (HIGH confidence)
- [MDN - Window.localStorage](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)
- [MDN - Storage.setItem()](https://developer.mozilla.org/en-US/docs/Web/API/Storage/setItem)
- [MDN - Storage.removeItem()](https://developer.mozilla.org/en-US/docs/Web/API/Storage/removeItem)
- [MDN - Storage quotas and eviction criteria](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria)
- [MDN - Date.now()](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/now)
- [MDN - Performance.now()](https://developer.mozilla.org/en-US/docs/Web/API/Performance/now)

### Secondary (MEDIUM confidence)
- [Meticulous.ai - JavaScript LocalStorage Complete Guide](https://www.meticulous.ai/blog/localstorage-complete-guide) - 2026 best practices
- [LogRocket - localStorage in JavaScript: A complete guide](https://blog.logrocket.com/localstorage-javascript-complete-guide/) - 2026 patterns
- [JavaScript.info - LocalStorage, sessionStorage](https://javascript.info/localstorage) - Modern reference
- [Gamedev.js - Using local storage for high scores and game progress](https://gamedevjs.com/articles/using-local-storage-for-high-scores-and-game-progress/) - Game-specific patterns
- [CodingBeast - LocalStorage Quota Exceeded Errors](https://codingbeast.org/localstorage-quota-exceeded-errors/) - Error handling
- [CSS-Tricks - Animating Number Counters](https://css-tricks.com/animating-number-counters/) - Animation patterns
- [Game UI Database - Results Screen](https://www.gameuidatabase.com/index.php?scrn=53) - UI patterns reference

### Tertiary (LOW confidence)
- Various CodePen examples for canvas text animation - useful for implementation ideas but not authoritative
- Medium articles on localStorage patterns - provide context but need verification with official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All native browser APIs, well-documented on MDN
- Architecture: HIGH - Existing codebase patterns verified, localStorage patterns from official docs
- Pitfalls: HIGH - Common issues documented in MDN and verified error handling guides

**Research date:** 2026-02-06
**Valid until:** 2026-03-06 (30 days - stable browser APIs, slow-changing domain)

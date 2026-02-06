# Research Summary: Tetris Twist v3.0

**Project:** Tetris Twist v3.0 Polish & Persistence
**Domain:** Browser-based game audio polish and persistence
**Researched:** 2026-02-06
**Confidence:** HIGH

## Executive Summary

Tetris Twist v3.0 adds audio polish and persistence features to an existing vanilla JS Tetris game with validated Web Audio API and localStorage patterns. The research confirms all four features require zero new dependencies and integrate cleanly with existing architecture. Combo pitch scaling extends existing OscillatorNode sound effects by passing combo count to frequency calculations. Background music uses procedural tone generation with continuous oscillators instead of audio files to maintain zero dependencies. Personal best tracking extends existing localStorage patterns from audio mute and key bindings to session stats. Key binding export/import uses standard File API with Blob download and FileReader upload patterns.

Critical finding: v3.0 must avoid 19 identified pitfalls, with highest severity around Web Audio parameter ramping, AudioBufferSourceNode lifecycle, localStorage error handling, and integration with existing mute state. The existing architecture is sound and all features are additive with no breaking changes needed. Implementation order matters: audio features first to validate parameter ramping patterns before persistence features leverage established localStorage patterns. Background music should come last due to higher complexity in state management and mute integration.

Risk mitigation centers on three areas: Web Audio API requires parameter ramping to prevent clicks, single AudioContext architecture to prevent memory leaks, and Safari compatibility requiring playbackRate fallback since detune is unsupported. localStorage requires comprehensive try-catch wrapping, schema versioning for data migration, and quota detection for incognito mode. Integration requires careful testing of mute state affecting both effects and music, personal best syncing with session stats, and key binding export using current in-memory state not stale localStorage values.

## Key Findings

### Recommended Stack

**Zero new dependencies required.** All features achievable with existing validated stack:

- **Web Audio API (OscillatorNode):** Combo pitch scaling via frequency.value modification, background music via continuous oscillator with low gain instead of audio files
- **localStorage:** Personal best persistence extends existing patterns from audio mute and keymap storage
- **File API (Blob, FileReader):** Key binding export uses URL.createObjectURL for download, FileReader.readAsText for upload
- **JSON:** Native serialization for stats and keybindings, no schema validation library needed

**What's NOT needed:**
- Audio libraries (Tone.js, Howler.js): 50-100KB overhead for capabilities already present
- Storage libraries (localForage, Dexie.js): Unnecessary IndexedDB complexity for <1KB data
- File libraries (FileSaver.js): 3KB dependency to replace 5 lines of native Blob code
- JSON validation libraries (ajv, joi): 5-line validation sufficient for simple schema

### Expected Features

**Table stakes (must-have):**
- Combo pitch scaling with linear frequency increase, capped at 10x to prevent painful high frequencies
- Background music looping seamlessly without gaps, with independent volume control from effects
- Personal best tracking for score, lines, level, date across sessions
- Key binding export to JSON file with validation on import

**Differentiators (should defer to post-v3.0):**
- Adaptive music changing based on game state: requires multiple audio tracks and complex state management
- Multiple personal best categories: feature creep, score metric sufficient for MVP
- Preset binding slots: nice-to-have, single export/import sufficient
- Tetris Effect-style music where player actions trigger musical elements: requires complex audio synthesis beyond procedural oscillators

**Anti-features (never implement):**
- Infinite pitch scaling: becomes ear-piercing at high combos
- Music with lyrics: distracting for puzzle games
- Global leaderboards: requires backend, out of scope
- Social sharing: adds complexity, not core to puzzle game loop

### Architecture Approach

**Integration strategy:** Extend existing modules, create zero new modules

**Module integration:**
- **audio.js:** Add combo parameter to playLineClearSound(combo), add startBackgroundMusic/stopBackgroundMusic/pauseBackgroundMusic/resumeBackgroundMusic functions, maintain single AudioContext with separate GainNodes for effects and music
- **stats.js:** Add loadPersonalBest/savePersonalBest/checkNewBest functions using localStorage key tetris_personal_best, integrate with existing session stats tracking
- **input.js:** Add exportKeymap/importKeymap/validateKeymapStructure functions, serialize existing keymap structure to JSON
- **admin.js + admin.html:** Add export/import buttons to Controls section, add separate music mute toggle in Audio section

**Critical architectural decisions:**
1. **Separate music mute toggle:** Use two localStorage keys (audio_muted, music_muted) for independent control
2. **Personal best scope:** Track score, lines, level, date with schema version for migration
3. **Combo pitch formula:** Linear with cap: 440Hz + (Math.min(combo, 10) * 40Hz)
4. **Export filename:** Branded "tetris_keybindings.json" without timestamp for simplicity

**Data flow patterns:**
- Combo pitch: main.js combo tracking → audio.js playLineClearSound(combo) → OscillatorNode frequency scaling
- Background music: main.js GameState transitions → audio.js music control functions → continuous OscillatorNode with loop management
- Personal best: main.js game over → stats.js checkNewBest → localStorage save if new record → render.js display comparison
- Key binding export/import: admin.html buttons → admin.js handlers → input.js serialization → Blob download or FileReader upload

### Critical Pitfalls

**Top 7 highest-severity pitfalls identified:**

1. **Parameter ramping for pitch changes:** Directly setting frequency.value causes clicking artifacts. Use exponentialRampToValueAtTime with 0.01-0.05s ramp duration to prevent discontinuities. This is critical to implement from start to avoid audio rewrites.

2. **AudioBufferSourceNode one-time use:** Cannot restart source nodes after stop(). Create new node for each playback. Store AudioBuffer, not source node. This breaks music restart if violated.

3. **Mute state integration:** New background music must respect existing localStorage audio_muted state. Connect music GainNode to same mute logic. High likelihood of missing this during initial integration.

4. **localStorage quota exceeded:** Wrap all setItem calls in try-catch for QuotaExceededError. Implement graceful fallback. Show user error message. Critical for incognito mode where quota is 0.

5. **JSON.parse corruption handling:** Wrap all JSON.parse calls in try-catch. Validate schema after parsing. Return default values on failure. Remove corrupted entries. Game crashes on startup if violated.

6. **Import file validation:** Validate MIME type, file size, JSON structure, and schema before applying. Sanitize all dynamic content. Reject oversized files. Security and stability critical.

7. **Safari detune property missing:** Safari doesn't support AudioBufferSourceNode.detune. Feature detection required with playbackRate fallback. High likelihood since development likely happens in Chrome.

**Other critical concerns:**
- Frequency range violations: Cap at Nyquist limit (sampleRate/2 * 0.9) to prevent aliasing
- MP3 loop gaps: Use OGG Vorbis for seamless looping or set custom loopStart/loopEnd
- Multiple AudioContext memory leaks: Use single context with separate GainNodes, not multiple contexts
- Blob URL memory leaks: Always URL.revokeObjectURL after export with 100ms timeout
- Schema versioning: Include version field in localStorage data for migration on updates

## Implications for Roadmap

**Suggested phase structure: 4 phases matching 4 features**

### Phase 1: Combo Pitch Scaling
**Rationale:** Simplest feature, extends existing audio.js, leverages validated combo tracking, zero dependencies, validates parameter ramping pattern for later audio work

**Delivers:** Real-time audio feedback that scales with player performance, immediate juice for existing combo system

**Features from FEATURES.md:**
- Linear frequency scaling with cap at 10x combo
- Applied to line clear sound effects
- Resets when combo breaks

**Pitfalls to avoid:**
- Pitfall 1: Use frequency.value not playbackRate to prevent duration coupling
- Pitfall 2: Implement exponentialRampToValueAtTime from start to prevent clicks
- Pitfall 18: Feature detection for Safari detune support with playbackRate fallback
- Pitfall 3: Cap frequency at Nyquist limit for high combos

**Complexity:** LOW

### Phase 2: Personal Best Tracking
**Rationale:** Independent of audio changes, extends existing stats.js patterns, validates localStorage error handling for later export feature, additive only with no gameplay changes

**Delivers:** Player engagement through progress tracking, retention mechanic (70% factor from research), clear milestone achievement feedback

**Features from FEATURES.md:**
- Persist high score, max lines, level, date across sessions
- Display comparison at game over
- "New record" notification when beaten

**Pitfalls to avoid:**
- Pitfall 8: Wrap localStorage.setItem in try-catch for QuotaExceededError
- Pitfall 9: Wrap JSON.parse in try-catch with schema validation
- Pitfall 11: Include schema version field for future migration
- Pitfall 16: Integrate checkNewBest with existing session stats, update both
- Pitfall 19: Detect localStorage availability for incognito mode with fallback

**Complexity:** LOW

### Phase 3: Key Binding Export/Import
**Rationale:** Extends existing input system, UI-only feature with no gameplay dependencies, validates File API patterns, completes v2.0 remapping story

**Delivers:** Portability of custom controls across installations, backup for settings, completes control customization feature set

**Features from FEATURES.md:**
- Export current key bindings to JSON file
- Import key bindings from JSON file with validation
- Error messages for invalid imports
- Works across browser sessions

**Pitfalls to avoid:**
- Pitfall 12: Call URL.revokeObjectURL after export to prevent memory leaks
- Pitfall 13: Sanitize filename with whitelist [a-zA-Z0-9-_.] before download
- Pitfall 14: Validate MIME type, file size, JSON structure, schema on import
- Pitfall 17: Export from current in-memory keymap, not stale localStorage

**Complexity:** LOW

### Phase 4: Background Music
**Rationale:** Most complex, requires new audio state management, depends on Phase 1 validating parameter ramping, test last to avoid breaking sound effects, procedural generation maintains zero dependencies

**Delivers:** Atmospheric enhancement, flow state support, audio polish that differentiates from basic Tetris clones

**Features from FEATURES.md:**
- Seamless looping procedural music using continuous OscillatorNode
- Independent volume control from sound effects
- Starts/stops with game session, pauses when game pauses
- Separate mute toggle from effects

**Pitfalls to avoid:**
- Pitfall 4: Create new source nodes for each playback, don't reuse stopped nodes
- Pitfall 5: Use procedural generation (OGG if static files added later) for seamless looping
- Pitfall 6: Share single AudioContext with effects, use separate GainNode for music
- Pitfall 7: Use audioContext.currentTime for all timing, not Date.now()
- Pitfall 15: Integrate music GainNode with existing mute state from localStorage

**Complexity:** MEDIUM

### Phase Ordering Rationale

**Why this order:**
1. **Audio features first (Phases 1 & 4):** Validates Web Audio patterns (parameter ramping, single context architecture) before persistence features
2. **Combo pitch before music:** Simpler audio feature validates ramping technique needed for more complex music state management
3. **Personal best before export:** Validates localStorage error handling patterns that export depends on
4. **Background music last:** Highest complexity, most integration points, benefits from validated patterns in earlier phases

**Dependencies:**
- Phase 1 validates parameter ramping needed for Phase 4 music
- Phase 2 validates localStorage error handling needed for Phase 3 export
- Phase 4 depends on Phase 1 audio patterns and Phase 2 localStorage patterns
- Phases 1-3 are independent and could parallelize if desired

### Research Flags

**Phases needing deeper research during planning:**
- **None.** All four features have well-documented patterns and clear implementation paths. Research confidence is HIGH across all areas.

**Phases with standard patterns (skip research):**
- **All four phases.** Stack research confirms no new dependencies needed. Features research identifies table stakes. Architecture research shows clean integration. Pitfalls research provides comprehensive risk mitigation.

**Validation needed during implementation:**
- Phase 1: Safari compatibility testing for detune fallback
- Phase 2: Data migration testing with v2.0 localStorage format
- Phase 3: Round-trip testing (export then import)
- Phase 4: Loop seamlessness testing and mute integration testing

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new dependencies, all APIs validated in v2.0 or standard browser features with universal support |
| Features | MEDIUM | Table stakes clear, but background music audio file sourcing deferred by using procedural generation instead |
| Architecture | HIGH | Clean integration with existing modules, no new modules needed, patterns match v1.0/v2.0 precedent |
| Pitfalls | HIGH | 19 pitfalls identified with authoritative sources (MDN, GitHub issues, 2025-2026 articles), severity assessed |
| Overall | HIGH | All features achievable with clear implementation paths, main risks identified with prevention strategies |

**Confidence factors:**
- Web Audio API patterns verified with MDN documentation and real-world GitHub issues
- localStorage patterns established in existing codebase (keymap, audio mute)
- File API is standard browser functionality with universal support
- Pitfall research corroborated with multiple authoritative sources
- Integration points well-defined in existing architecture

**Gaps identified:**
- Background music composition: Procedural generation chosen to maintain zero dependencies, but no specific melody designed yet. This is implementation detail, not research gap.
- Combo pitch tuning: Formula provided (440Hz + Math.min(combo, 10) * 40Hz) but exact step size may need playtesting adjustment. Not a blocker.
- Personal best display UI: render.js extension needed but pattern straightforward from existing session stats display.

**Areas for validation during implementation:**
- Safari compatibility for pitch scaling (detune property unsupported, needs playbackRate fallback)
- Loop seamlessness for background music (use procedural generation to avoid MP3 gap issues)
- localStorage quota handling in incognito mode (feature detection and graceful fallback)
- Multi-tab scenarios for personal best updates (low priority for single-player game)

## Sources

**Stack Research:**
- [MDN Web Audio API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [MDN OscillatorNode.frequency](https://developer.mozilla.org/en-US/docs/Web/API/OscillatorNode/frequency)
- [MDN AudioBufferSourceNode.loop](https://developer.mozilla.org/en-US/docs/Web/API/AudioBufferSourceNode/loop)
- [MDN File API](https://developer.mozilla.org/en-US/docs/Web/API/File_API)
- [localStorage Complete Guide - Meticulous](https://www.meticulous.ai/blog/localstorage-complete-guide)
- [localStorage Best Practices 2026](https://copyprogramming.com/howto/javascript-how-ot-keep-local-storage-on-refresh)
- [json-porter GitHub](https://github.com/markgab/json-porter)

**Features Research:**
- [Game Audio Analysis - Tetris Effect](https://www.gamedeveloper.com/audio/game-audio-analysis---tetris-effect)
- [Tetris Effect - Enhancing gameplay with synesthesia](https://www.nicholassinger.com/blog/tetriseffect)
- [Pitch shifting in Web Audio API](https://zpl.fi/pitch-shifting-in-web-audio-api/)
- [A Game Developer's Guide to Gaming Background Music](https://www.dl-sounds.com/a-game-developers-guide-to-gaming-background-music/)
- [Design With Music In Mind: A Guide to Adaptive Audio for Game Designers](https://www.gamedeveloper.com/audio/design-with-music-in-mind-a-guide-to-adaptive-audio-for-game-designers)
- [Audio for Web games - MDN](https://developer.mozilla.org/en-US/docs/Games/Techniques/Audio_for_Web_Games)
- [Fitting the pieces: Decoding trends and behaviors of modern puzzle gamers](https://business.mistplay.com/resources/puzzle-game-trends)
- [Using local storage for high scores and game progress](https://gamedevjs.com/articles/using-local-storage-for-high-scores-and-game-progress/)

**Architecture Research:**
- Tetris Twist v2.0 existing codebase patterns

**Pitfalls Research:**
- [Web Audio: the ugly click and the human ear](http://alemangui.github.io/ramp-to-value)
- [Web Audio API performance and debugging notes](https://padenot.github.io/web-audio-perf/)
- [Web Audio FAQ - Chrome for Developers](https://developer.chrome.com/blog/web-audio-faq)
- [Sounds fun - JakeArchibald.com](https://jakearchibald.com/2016/sounds-fun/)
- [Using the Web Audio API - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Using_Web_Audio_API)
- [Storage quotas and eviction criteria - MDN](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria)
- [Understanding and Resolving LocalStorage Quota Exceeded Errors](https://medium.com/@zahidbashirkhan/understanding-and-resolving-localstorage-quota-exceeded-errors-5ce72b1d577a)
- [Stop Using JSON.parse(localStorage.getItem(…)) Without This Check](https://medium.com/devmap/stop-using-json-parse-localstorage-getitem-without-this-check-94cd034e092e)
- [JavaScript concurrency and locking the HTML5 localStorage](https://balpha.de/2012/03/javascript-concurrency-and-locking-the-html5-localstorage/)
- [Programmatically downloading files in the browser](https://blog.logrocket.com/programmatically-downloading-files-browser/)
- [How to upload and process a JSON file with vanilla JS](https://gomakethings.com/how-to-upload-and-process-a-json-file-with-vanilla-js/)
- [Pitch shifting in Web Audio API](https://zpl.fi/pitch-shifting-in-web-audio-api/)

---
*Research completed: 2026-02-06*
*Ready for roadmap: yes*

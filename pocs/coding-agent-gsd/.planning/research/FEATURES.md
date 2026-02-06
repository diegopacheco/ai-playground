# Features Research: Tetris Twist v3.0

**Researched:** 2026-02-06
**Focus:** Audio polish and persistence features
**Confidence:** MEDIUM

## Feature Categories

### Combo Pitch Scaling

**Table Stakes:**
- Pitch increases proportionally with combo counter (1x = base pitch, 2x = higher, 3x = even higher)
- Uses existing combo counter (already tracked in game state)
- Applied to line clear sound effects
- Scales using semitone ratio (multiply by 2^(1/12) per step) for musical correctness
- Pitch range limited to prevent unpleasant frequencies (typically 1-2 octaves max)
- Resets when combo breaks

**Differentiators:**
- Different pitch patterns per line clear type (single vs double vs triple vs Tetris)
- Pitch feedback on piece placement that varies by combo level
- Smooth pitch transitions rather than discrete jumps
- Visual feedback synchronized with pitch changes
- Tetris Effect-style dynamic sound where each move adds pitched vocal chops to create player-driven music
- Different timbres (oscillator types) at different combo levels

**Anti-Features:**
- Infinite pitch scaling (becomes ear-piercing)
- Pitch scaling without combo visual feedback (confuses players about why sound changed)
- Pitch scaling on game over sound (jarring, not rewarding)
- Complex pitch patterns that distract from gameplay

**Integration:**
- Hooks into existing playLineClearSound() and playTetrisSound() functions
- Uses existing combo counter from game state
- Respects existing mute toggle and localStorage persistence
- Works with Web Audio API OscillatorNode already in use

**Complexity:** LOW
- Simple playbackRate or frequency multiplication
- Existing audio infrastructure supports it
- Combo counter already tracked

### Background Music

**Table Stakes:**
- Seamless looping without noticeable restart seams
- Independent volume control from sound effects
- Starts/stops with game session
- Pauses when game pauses
- Respects global mute toggle
- Single background track is sufficient
- Music should be medium-paced, gentle, not too percussive
- Does not overpower sound effects

**Differentiators:**
- Adaptive music that changes based on game state (level, speed, intensity)
- Music tempo synced to piece fall speed
- Layered tracks that add/remove instruments based on combo or level
- Multiple tracks with theme-based selection
- Dynamic transitions between musical sections
- Tetris Effect-style music where player actions trigger musical elements
- Cross-fade between tracks on theme change

**Anti-Features:**
- Music with lyrics (distracting for puzzle games)
- Overly complex, attention-demanding compositions
- Music louder than sound effects
- Auto-play before user interaction (browsers block this anyway)
- High-percussive or aggressive music (increases stress instead of flow state)
- Music that disconnects from gameplay (causes player distraction)

**Integration:**
- Web Audio API AudioBufferSourceNode for looping
- Separate gain node from sound effects for independent volume
- Hooks into game state (pause, resume, game over)
- Respects existing mute toggle
- Loads on game start after user interaction (browser autoplay policy)

**Complexity:** MEDIUM
- Requires audio file loading and buffering
- Loop point management for seamless playback
- State synchronization (pause, resume, game over)
- Separate volume mixing

### Personal Best Tracking

**Table Stakes:**
- Track high score across sessions (localStorage)
- Track max combo across sessions
- Display during gameplay if current session beats personal best
- Display at game over with comparison to current session
- Metrics tracked: score, lines, max combo, time survived
- Persists across browser sessions
- Clear/reset option available

**Differentiators:**
- Track multiple categories (best score, best lines, best PPS, best APM, best efficiency)
- Date/timestamp of when personal best was achieved
- Performance trends (improvement over time)
- "New record" visual/audio celebration when beaten
- Separate bests for different difficulty levels or themes
- Export personal bests with key bindings
- Cloud sync across devices

**Anti-Features:**
- Global leaderboards (out of scope, requires backend)
- Social sharing (adds complexity, not core to puzzle game loop)
- Achievements/badges system (feature creep)
- Analytics tracking to external services (privacy concerns)
- Persistent login/account requirement (contradicts session-only design)

**Integration:**
- Uses existing stats tracking (score, lines, PPS, APM, efficiency, max combo)
- Stored in localStorage alongside mute preference and key bindings
- Display hooks into existing session stats UI
- Loads on game initialization
- Updates on game over if records beaten

**Complexity:** LOW
- Simple localStorage read/write
- Comparison logic on game over
- Stats already tracked, just need persistence
- UI already displays stats

### Key Binding Export/Import

**Table Stakes:**
- Export current key bindings to JSON file
- Import key bindings from JSON file
- File validation (reject malformed JSON, missing required actions)
- Error messages for invalid imports
- Overwrites existing bindings on successful import
- Works across browser sessions (import saved bindings on different machine)
- Standard JSON format with action-to-keycode mapping

**Differentiators:**
- Export includes metadata (date exported, game version, player notes)
- Import validates for conflicts before applying
- Preview bindings before applying import
- Merge mode (import only specific actions, keep others)
- Multiple preset slots (save/load 3-5 different binding sets)
- Share bindings via URL/clipboard (encode as base64)
- Backup/restore with personal bests bundled together

**Anti-Features:**
- Cloud sync (requires backend)
- Binding profiles synced across devices automatically (complexity)
- Import from other Tetris games (format incompatibility)
- Automatic conflict resolution without user confirmation (loses user control)
- Binary format (reduces portability and debuggability)

**Integration:**
- Uses existing keymap structure in input.js
- Hooks into existing localStorage save/load mechanism
- Triggers existing conflict detection on import
- Export button in admin panel Controls section
- Import via file input or drag-and-drop
- Broadcasts keymap change via existing BroadcastChannel

**Complexity:** LOW
- Serialize/deserialize existing keymap object
- File download via Blob and URL.createObjectURL
- File upload via FileReader API
- Validation reuses existing conflict detection
- Format already JSON in localStorage

## Feature Interactions

**Combo Pitch + Background Music:**
- Sound effects layer on top of music
- Both respect mute toggle together
- Pitch scaling creates polyrhythmic interplay with steady background music
- Music gain adjusted lower to ensure pitch-scaled effects are audible

**Personal Best + Stats:**
- Personal best compares against existing session stats
- Same metrics tracked, just persisted
- UI shows both current session and all-time best side-by-side

**Key Bindings + Personal Best:**
- Export could bundle both (user preference portability)
- Import validates keys work on current keyboard layout
- Personal bests remain valid across different control schemes

**Audio Polish + Real-time Admin:**
- Mute toggle already syncs across tabs
- Background music playback state could sync
- Admin panel could control music volume separately

## User Expectations

Based on research, modern puzzle game players in 2026 expect:

**Audio Feedback:**
- Dynamic, player-driven music creation where actions influence sound
- Pitch and musical elements responding to gameplay performance
- Synesthetic experience tying audio to visual feedback
- Music that enhances flow state without distracting
- Separate control over music vs effects

**Progress Tracking:**
- Personal best is a core engagement mechanic (70% retention factor)
- Progress tracking with clear metrics (time, performance stats)
- Daily engagement incentives (though streaks not in v3.0 scope)
- AI-driven personalization emerging trend (out of scope)

**Control Customization:**
- Full keyboard remapping is expected in modern games
- Import/export for portability across installations
- JSON format standard for game settings
- Clear conflict detection and validation

**Quality Expectations:**
- Seamless audio loops without pops or clicks
- Music that doesn't overpower gameplay
- Performance metrics that help players improve
- Settings that persist across sessions

## MVP Recommendation

For v3.0 MVP, prioritize in this order:

1. **Combo pitch scaling** - Builds on existing audio, immediate feedback impact, LOW complexity
2. **Personal best tracking** - Core engagement mechanic, LOW complexity, high retention value
3. **Key binding export/import** - Completes the remapping feature from v2.0, LOW complexity
4. **Background music** - MEDIUM complexity, defer if time-constrained

Rationale:
- All three LOW complexity features deliver high value quickly
- Background music requires audio file sourcing and more integration work
- Combo pitch scaling enhances existing combo system (v2.0 feature)
- Personal best tracking addresses 70% retention factor (puzzle game player expectation)
- Key binding export completes control customization story

Defer to post-v3.0:
- Adaptive music (theme/level-based) - Requires multiple audio files and complex state management
- Multiple personal best categories - Feature creep, core metric (score) sufficient for v3.0
- Preset binding slots - Nice to have, single export/import sufficient
- Music tempo sync - Complex implementation, not table stakes

## Implementation Dependencies

**Combo Pitch Scaling requires:**
- Existing: combo counter, audio system, OscillatorNode
- New: pitch calculation function, combo-to-pitch mapping

**Background Music requires:**
- Existing: Web Audio API context, mute toggle, game state
- New: audio file, AudioBufferSourceNode, loop management, separate gain node

**Personal Best Tracking requires:**
- Existing: stats tracking, localStorage, session stats UI
- New: persistence logic, comparison on game over, best stats structure

**Key Binding Export/Import requires:**
- Existing: keymap structure, localStorage, conflict detection
- New: JSON serialization, file download/upload, validation logic

## Edge Cases to Handle

**Combo Pitch Scaling:**
- Combo counter resets mid-game (pitch should reset smoothly)
- Very high combos (cap pitch at reasonable frequency)
- Mute toggled during pitch-scaled sound (stop oscillator immediately)
- Multiple line clears happening rapidly (queue sounds vs overlap)

**Background Music:**
- User interaction required before playback (browser autoplay policy)
- Music file fails to load (graceful degradation, no crash)
- Pause during music playback (resume from same position vs restart)
- Game over during music (fade out vs hard stop)
- Theme change mid-game (cross-fade vs immediate switch vs restart track)

**Personal Best Tracking:**
- First time playing (no previous best exists)
- Tied with previous best (still show "matched personal best")
- localStorage quota exceeded (handle gracefully, warn user)
- Corrupted localStorage data (validate and reset if invalid)
- Multiple tabs with different personal bests (last write wins, could show warning)

**Key Binding Export/Import:**
- Malformed JSON file (show error message, don't crash)
- JSON with missing actions (reject import, list missing keys)
- JSON with extra unknown actions (ignore extras, import known ones)
- Key codes that don't exist on current keyboard (validation and warning)
- Import file from future game version (forward compatibility check)
- File too large (size validation before parsing)

## Technical Constraints

**Web Audio API:**
- Requires user interaction before creating AudioContext (handle suspended state)
- OscillatorNode can't be restarted once stopped (create new node each time)
- AudioBufferSourceNode can only play once (create new for each playback)
- Exponential ramp requires value > 0 (use 0.01 instead of 0)

**localStorage:**
- 5-10 MB limit per domain (unlikely to hit with simple key-value data)
- String-only storage (JSON.stringify/parse required)
- Synchronous API (could block UI on large operations)
- Domain-specific (won't sync across different domains)

**File API:**
- Browser security prevents auto-download without user interaction
- FileReader API is asynchronous (handle loading state)
- MIME type validation not guaranteed (validate content, not just extension)

**BroadcastChannel:**
- Same-origin only (already project constraint)
- No persistence (messages not queued if no listeners)
- Not supported in Safari < 15.4 (check compatibility or degrade gracefully)

## Confidence Assessment

| Feature | Confidence | Rationale |
|---------|-----------|-----------|
| Combo Pitch Scaling | HIGH | Web Audio API documentation clear, pattern well-established in games, existing audio infrastructure ready |
| Background Music | MEDIUM | Implementation clear, but seamless looping and state management has edge cases, audio file sourcing required |
| Personal Best Tracking | HIGH | localStorage well-documented, pattern standard in web games, existing stats system ready |
| Key Binding Export/Import | HIGH | File API well-documented, JSON format standard, existing keymap structure supports it |

Overall confidence: MEDIUM
- All features have clear implementation paths
- Edge cases identified and addressable
- Dependencies on existing systems minimal
- Main uncertainty: background music file sourcing and seamless loop quality

## Sources

**Tetris Effect Audio Design:**
- [Game Audio Analysis - Tetris Effect](https://www.gamedeveloper.com/audio/game-audio-analysis---tetris-effect)
- [Tetris Effect - Enhancing gameplay with synesthesia](https://www.nicholassinger.com/blog/tetriseffect)

**Web Audio Implementation:**
- [Pitch shifting in Web Audio API](https://zpl.fi/pitch-shifting-in-web-audio-api/)
- [Controlling Frequency and Pitch](https://teropa.info/blog/2016/08/10/frequency-and-pitch.html)
- [The Power of Pitch Shifting](https://www.gamedeveloper.com/audio/the-power-of-pitch-shifting)

**Background Music Best Practices:**
- [A Game Developer's Guide to Gaming Background Music](https://www.dl-sounds.com/a-game-developers-guide-to-gaming-background-music/)
- [Design With Music In Mind: A Guide to Adaptive Audio for Game Designers](https://www.gamedeveloper.com/audio/design-with-music-in-mind-a-guide-to-adaptive-audio-for-game-designers)
- [Audio for Web games - MDN](https://developer.mozilla.org/en-US/docs/Games/Techniques/Audio_for_Web_Games)

**Seamless Looping:**
- [Looping Music in Unity](https://andrewmushel.com/articles/looping-music-in-unity/)

**Personal Best & Player Engagement:**
- [Fitting the pieces: Decoding trends and behaviors of modern puzzle gamers](https://business.mistplay.com/resources/puzzle-game-trends)
- [22 metrics all game developers should know by heart](https://www.gameanalytics.com/blog/metrics-all-game-developers-should-know)

**localStorage for Game Data:**
- [Using local storage for high scores and game progress](https://gamedevjs.com/articles/using-local-storage-for-high-scores-and-game-progress/)
- [How to Save High Scores in Local Storage](https://michael-karen.medium.com/how-to-save-high-scores-in-local-storage-7860baca9d68)
- [HTML5 Games Development: Using Local Storage to Store Game Data](https://hub.packtpub.com/html5-games-development-using-local-storage-store-game-data/)

**Key Binding Import/Export:**
- [Export key bindings as JSON file? - Wildfire Games Community Forums](https://wildfiregames.com/forum/topic/40221-export-key-bindings-as-json-file-useful-to-create-and-print-out-nice-keyboard-templates/)
- [keybindings.json - GitHub Gist](https://gist.github.com/rwu823/bb0e3f983ca405281579757f4ee8813b)

**Validation & Error Handling:**
- [Data Validation and Error Handling Best Practices](https://echobind.com/post/data-validation-error-handling-best-practices)

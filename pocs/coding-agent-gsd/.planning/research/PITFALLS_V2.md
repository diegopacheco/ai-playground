# Pitfalls Research: Tetris Twist v2.0

**Domain:** Advanced Tetris features (T-spin, combo, audio, key remapping)
**Researched:** 2026-02-03
**Confidence:** MEDIUM (WebSearch verified with multiple sources)

This document catalogs common mistakes when adding T-spin detection, combo scoring, audio, and key remapping to an existing Tetris game. Focus is on pitfalls specific to adding these features without breaking working core mechanics.

## T-Spin Detection Pitfalls

### Critical Pitfall 1: Wall/Floor Corner Detection Inconsistencies

**What goes wrong:**
T-spin detection fails when T-piece is against playfield boundaries. In some implementations, walls and floor are considered occupied for corner detection, in others they are not. This causes valid T-spins against walls to not register.

**Why it happens:**
Corner detection algorithms check if 3 of 4 diagonal corners around the T-piece center are occupied. Developers forget that the playfield boundary is different from blocks.

**Consequences:**
- Valid T-spins at walls/floor not recognized
- Player frustration when expected scoring does not occur
- Inconsistent behavior between left wall, right wall, and floor

**Prevention:**
Explicitly define boundary behavior: treat playfield edges as occupied for corner detection. Test T-spins specifically at x=0, x=width-1, and y=height-1.

**Detection:**
Test cases failing at boundaries but passing in center field. Player reports of T-spins not registering near walls.

### Critical Pitfall 2: Rotation-Triggered Realignment Breaking Detection

**What goes wrong:**
When T-piece rotates with flat side vertical near bottom, collision resolution pushes it down one row. This changes the center of rotation and prevents T-spin detection for that specific rotation scenario.

**Why it happens:**
Collision detection runs after rotation, adjusting position. T-spin detection then checks the wrong position for corner occupancy.

**Consequences:**
Specific rotation sequences fail T-spin detection while others work. Creates perception of buggy/inconsistent mechanics.

**Prevention:**
Check T-spin conditions BEFORE applying position adjustments. Store pre-adjustment position for corner checks.

**Detection:**
T-spins work when rotating from horizontal but fail when rotating from vertical near bottom. Off-by-one errors in corner detection.

### Critical Pitfall 3: T-Spin Mini vs Full T-Spin Classification Errors

**What goes wrong:**
Incorrect distinction between Mini and Full T-spins, especially with wall kicks. Some valid Full T-spins classified as Mini or not recognized at all.

**Why it happens:**
Different Tetris implementations use different rules. Tetra-X kickset incorrectly defines at least one T-Spin Mini scenario as full T-Spin. Triple T-spins may not register at all in some rotation scenarios.

**Consequences:**
- Wrong points awarded (Mini = 2 lines cleared worth, Full = 4 lines)
- Triple T-spins not recognized (Tetris Zone, Tetris Evolution documented bugs)
- Competitive players notice scoring inconsistencies

**Prevention:**
Implement standard Guideline SRS wall kick rules. Explicitly test all 4 rotation states. Verify Triple T-spin detection works.

**Detection:**
Scoring values incorrect. Players report "that should have been a T-spin" scenarios.

### Moderate Pitfall 4: False Positive T-Spin Detection

**What goes wrong:**
T-spin detected when piece was simply dropped into position without rotation, awarding unearned points.

**Consequences:**
Scoring inflation, competitive imbalance.

**Prevention:**
Track "last action before lock." T-spin only valid if last action was rotation.

**Detection:**
Unexpectedly high scores. T-spin counter increasing without actual twisting maneuvers.

### Moderate Pitfall 5: Breaking Existing Rotation When Adding T-Spin Detection

**What goes wrong:**
Adding T-spin corner detection interferes with existing SRS wall kick implementation, causing pieces to get stuck or rotate incorrectly.

**Why it happens:**
T-spin detection and rotation use overlapping logic. Developer modifies rotation code to add detection, breaking existing functionality.

**Consequences:**
Working rotation system now buggy. All pieces affected, not just T-piece.

**Prevention:**
Keep T-spin detection completely separate from rotation logic. Detection should be read-only check after rotation completes.

**Detection:**
Pieces no longer rotate correctly after T-spin detection added. Wall kicks stop working.

## Combo System Pitfalls

### Critical Pitfall 6: Off-By-One Errors in Combo Lookup Tables

**What goes wrong:**
Combo multiplier lookup tables have incorrect indices, causing combos of 9-10 to get same multiplier as combo 8, or Tetrises to receive wrong bonus.

**Why it happens:**
Array indexing error (0-based vs 1-based). TGM series had documented bugs with this exact issue.

**Consequences:**
- Incorrect scoring at high combo counts
- Tetris line clears receiving nerfed bonuses
- Players notice scoring inconsistencies at specific combo thresholds

**Prevention:**
Unit test every combo count from 0-20. Verify lookup table length matches maximum expected combo.

**Detection:**
Scoring stops increasing at specific combo counts. Console shows array index out of bounds errors.

### Critical Pitfall 7: Level Multiplier Applied at Wrong Time

**What goes wrong:**
When line clear increases level, multiplier applied for NEXT level instead of current level. Puyo Puyo Tetris had this exact bug.

**Why it happens:**
Level increment happens before score calculation. Developer uses updated level value instead of pre-increment value.

**Consequences:**
Scoring inconsistent at level boundaries. Players gain extra points when leveling up.

**Prevention:**
Calculate score with current level, then increment level. Or store previous level for score calculation.

**Detection:**
Unusually high scores when clearing lines that trigger level-up. Score jumps at level boundaries.

### Moderate Pitfall 8: Combo Counter Not Resetting

**What goes wrong:**
Combo counter persists across game resets, or does not reset when no lines cleared.

**Consequences:**
Combo bonus continues into new game. Scoring becomes meaningless.

**Prevention:**
Reset combo counter on game start and after any turn where zero lines cleared.

**Detection:**
Combo counter showing values immediately at game start. Combo never breaks.

### Moderate Pitfall 9: Combo Display Lag/Desync

**What goes wrong:**
Combo counter display updates before or after actual combo calculation, showing wrong value during scoring.

**Why it happens:**
UI update timing separate from game logic timing.

**Consequences:**
Displayed combo count does not match actual score awarded. Player confusion.

**Prevention:**
Update combo display in same game loop tick as combo calculation. Use single source of truth.

**Detection:**
Visual combo counter shows "5" but score awarded as if combo was "4".

### Minor Pitfall 10: Combo Notification Spam

**What goes wrong:**
Every combo increment triggers notification/animation, causing visual clutter and performance issues at high combos.

**Consequences:**
Screen filled with overlapping animations. Frame rate drops. Player cannot see playfield.

**Prevention:**
Throttle combo notifications. Show milestones (5, 10, 15) instead of every increment, or replace previous notification.

**Detection:**
Multiple "COMBO x5", "COMBO x6", "COMBO x7" messages stacked on screen simultaneously.

## Audio Pitfalls

### Critical Pitfall 11: Autoplay Policy Violation

**What goes wrong:**
Game attempts to play audio on page load before user interaction. Audio silently fails. AudioContext stuck in suspended state.

**Why it happens:**
Chrome 71+ blocks Web Audio API autoplay. Developer assumes audio can start immediately. AudioContext created before user gesture remains suspended.

**Consequences:**
- No sound throughout entire game session
- Silent failure with no user feedback
- Different behavior across browsers

**Prevention:**
Do not create AudioContext until first user interaction (click to start). Check context.state and call context.resume() after user gesture. Show UI indicator if audio blocked.

**Detection:**
Audio works in development but fails in production. console.log shows AudioContext state = "suspended".

### Critical Pitfall 12: Audio Memory Leaks

**What goes wrong:**
Sound effects create new Howler.js Howl instances on every play without cleanup. Memory grows 1GB per hour, causing crashes.

**Why it happens:**
Howler.js .unload() method does not properly release memory. Developer creates new instances in game loop instead of reusing.

**Consequences:**
- Browser tab crashes after extended play (4-5 hours)
- Frame stutters during garbage collection
- Mobile devices crash faster due to memory limits

**Prevention:**
Use object pooling for sound effects. Create Howl instances once at game start, reuse via .play(). Implement pooling strategy to avoid garbage collection overhead.

**Detection:**
Browser memory profiler shows growing heap. Performance degrades over time. Crashes during long sessions.

### Critical Pitfall 13: Large Uncompressed Audio Files

**What goes wrong:**
Audio files are WAV or uncompressed, causing long loading times and high memory usage.

**Why it happens:**
Developer exports audio in lossless format without optimization.

**Consequences:**
- Loading screen takes 10+ seconds
- High memory usage (100MB+ for audio alone)
- Performance issues on mobile/limited resource devices
- Reduced frame rates when loading audio

**Prevention:**
Use compressed formats (OGG, MP3). Keep sound effects under 50KB each. Stream music files instead of preloading. Target total audio assets under 5MB.

**Detection:**
Long initial load time. Network tab shows multi-megabyte audio downloads. Memory profiler shows excessive audio buffer allocation.

### Moderate Pitfall 14: Audio Causing Game Loop Lag

**What goes wrong:**
Playing multiple simultaneous sound effects (piece rotation, movement, lock) causes frame rate drops.

**Why it happens:**
Audio processing happens on main thread. Too many concurrent sounds overwhelm CPU. Elder Scrolls IV Oblivion had this exact issue with footstep sounds.

**Consequences:**
Game stutters during intense moments. Input feels unresponsive.

**Prevention:**
Limit concurrent sounds (max 8-12 simultaneous). Implement audio culling to prioritize important sounds. Use Web Audio API (which runs on separate thread) instead of HTML5 Audio.

**Detection:**
Frame rate drops when multiple pieces lock quickly. Performance profiler shows audio processing spikes.

### Moderate Pitfall 15: Mobile Audio Restrictions

**What goes wrong:**
Audio works on desktop but fails on mobile. Different behavior between iOS and Android.

**Why it happens:**
Mobile browsers have stricter autoplay policies. iOS Safari requires separate user gesture per audio element.

**Consequences:**
Silent game on mobile devices. First sound plays, subsequent sounds fail.

**Prevention:**
Test on actual mobile devices early. Ensure AudioContext created and resumed on first touch/tap. Consider mobile-specific audio loading strategy.

**Detection:**
Desktop works perfectly, mobile has no audio. Only first sound effect plays on iOS.

### Minor Pitfall 16: Audio and Visual Desync

**What goes wrong:**
Sound effect plays before/after visual feedback (piece lock sound plays while piece still falling).

**Consequences:**
Perceived lag. Game feels unpolished.

**Prevention:**
Trigger audio and visual updates in same game loop tick. Use requestAnimationFrame for synchronized timing.

**Detection:**
Audio plays noticeably before/after visual action. Players report timing feels off.

## Key Remapping Pitfalls

### Critical Pitfall 17: Remapped Keys Not Persisting

**What goes wrong:**
Player remaps controls, plays game, refreshes page. Settings lost.

**Why it happens:**
Key mappings stored in memory only, not localStorage. Or localStorage save fails silently (QuotaExceededError, SecurityError).

**Consequences:**
Players must reconfigure controls every session. Abandonment due to frustration.

**Prevention:**
Save to localStorage on every config change. Wrap in try/catch for QuotaExceededError. Handle private browsing mode (localStorage may be disabled). Provide export/import config as fallback.

**Detection:**
Settings reset on page refresh. Console shows storage errors. Users complain about lost settings.

### Critical Pitfall 18: Key Conflicts Not Detected

**What goes wrong:**
Player maps both "rotate left" and "rotate right" to same key. Or maps game action to browser shortcut (Ctrl+W closes tab).

**Why it happens:**
No validation when accepting key input. Developer assumes players will not create conflicts.

**Consequences:**
- Actions do not trigger (conflict with existing binding)
- Browser shortcuts triggered instead of game actions
- Tab closes mid-game (Ctrl+W)
- Print dialog opens (Ctrl+P)

**Prevention:**
Detect conflicts before saving. Show warning if key already bound. Blacklist browser shortcut keys (Ctrl+W, Ctrl+T, F5, etc). Use event.preventDefault() for all game keys.

**Detection:**
Multiple actions bound to same key. Browser shortcuts interfering with gameplay.

### Moderate Pitfall 19: Key Remapping Breaks Existing Gameplay

**What goes wrong:**
Adding remapping system changes how keyboard events are handled, causing existing controls to stop working.

**Why it happens:**
Developer replaces hardcoded key handling with new system but introduces bugs in event handling logic.

**Consequences:**
Working game now has broken controls. All players affected, not just those who remap.

**Prevention:**
Keep default key handling as fallback. Test default controls extensively after adding remapping. Use feature flag to isolate new code.

**Detection:**
Controls stop working after remapping feature added. Both default and custom keys affected.

### Moderate Pitfall 20: Special Keys Not Handled

**What goes wrong:**
Player tries to bind numpad keys, function keys, or media keys. Game does not recognize them.

**Why it happens:**
KeyboardEvent.key vs KeyboardEvent.code confusion. Different values for numpad vs main keyboard numbers.

**Consequences:**
Some keys cannot be bound. Numpad users cannot use preferred controls.

**Prevention:**
Use KeyboardEvent.code (physical key location) instead of KeyboardEvent.key (character produced). Test with numpad, function keys, international keyboards.

**Detection:**
Numpad keys do not register during rebinding. F1-F12 keys fail to bind.

### Moderate Pitfall 21: No Visual Feedback During Rebinding

**What goes wrong:**
Player clicks "remap key" but no indication that game is waiting for input. Player confused about what to do.

**Consequences:**
Poor UX. Players do not understand remapping interface.

**Prevention:**
Clear visual state: "Press any key for [action]...". Show currently pressed key before confirming. Allow ESC to cancel.

**Detection:**
User testing shows confusion during rebinding. Players clicking repeatedly instead of pressing new key.

### Minor Pitfall 22: Can't Unbind or Reset to Default

**What goes wrong:**
Player remaps controls poorly, cannot undo changes.

**Consequences:**
Game becomes unplayable. Player must clear localStorage manually or reinstall.

**Prevention:**
Provide "Reset to Default" button. Allow unbinding individual keys. Confirm before applying changes.

**Detection:**
Support requests asking how to reset controls. Players unable to fix bad configurations.

## Session Statistics Pitfalls

### Moderate Pitfall 23: Statistics Not Persisting Correctly

**What goes wrong:**
Session stats (games played, high score, total lines cleared) stored in sessionStorage instead of localStorage. Lost when tab closes.

**Why it happens:**
Developer confuses sessionStorage (tab-scoped, cleared on close) with localStorage (persistent).

**Consequences:**
Players lose all progress history when closing browser.

**Prevention:**
Use localStorage for persistent stats. Use sessionStorage only for current session temp data. Serialize objects to JSON before storing.

**Detection:**
Stats reset when reopening game. Only work within single browser session.

### Moderate Pitfall 24: Storage Quota Exceeded

**What goes wrong:**
Game stores unlimited history, eventually hits 5-10MB localStorage limit. QuotaExceededError thrown, stats stop saving.

**Why it happens:**
No limit on statistics history. Each game adds more data without cleanup.

**Consequences:**
Silent failure. Stats appear to save but do not. Older browsers more affected.

**Prevention:**
Limit history (keep last 100 games, not all games). Implement rolling window. Catch QuotaExceededError and prune old data.

**Detection:**
Console shows QuotaExceededError. Stats stop updating after playing many games.

### Minor Pitfall 25: Statistics Display Lag

**What goes wrong:**
Stats update at end of game, but computation blocks UI, causing visible freeze.

**Why it happens:**
Synchronous calculation of percentages, averages over large dataset on main thread.

**Consequences:**
UI freezes for 500ms-2s when viewing stats. Poor perceived performance.

**Prevention:**
Pre-calculate running averages instead of recomputing. Use Web Workers for heavy computation. Throttle updates.

**Detection:**
Visible freeze when opening statistics panel. Performance profiler shows long-running computation.

## Theme System Pitfalls

### Moderate Pitfall 26: Flash of Unthemed Content (FOUC)

**What goes wrong:**
Page loads with default theme, then flickers to user's saved theme after JavaScript loads.

**Why it happens:**
Theme script loads after CSS. localStorage read happens after initial paint.

**Consequences:**
Visual flash on every page load. Unprofessional appearance. Jarring for dark mode users.

**Prevention:**
Load theme preference in blocking inline script in <head>. Set theme class before first paint. Or use CSS variables with system preference default.

**Detection:**
Visible theme flash on page load. More noticeable with slow connections.

### Moderate Pitfall 27: Theme Switch Causes Performance Issues

**What goes wrong:**
Switching theme loads entire separate stylesheet, causing layout recalculation and frame drops.

**Why it happens:**
Multiple full stylesheets instead of CSS variables. Browser must reparse and reapply all styles.

**Consequences:**
Theme switch takes 500ms+ with visible lag. Animations stutter during switch.

**Prevention:**
Use CSS custom properties (variables) for theme colors. Only change variables, not entire stylesheet. Single stylesheet with variable overrides.

**Detection:**
Visible lag when clicking theme switcher. Performance profiler shows long layout recalculation.

### Minor Pitfall 28: Theme Not Respecting System Preference

**What goes wrong:**
Game always loads light theme, ignoring user's OS dark mode setting.

**Consequences:**
Dark mode users get blinded by white screen. Poor UX.

**Prevention:**
Check prefers-color-scheme media query as default. Override with saved preference if exists.

**Detection:**
Game ignores OS dark mode. Always shows light theme on first load.

## Integration Pitfalls

### Critical Pitfall 29: New Features Break Existing Core Mechanics

**What goes wrong:**
Adding T-spin detection modifies rotation logic, causing all pieces to rotate incorrectly. Or audio system interferes with game loop timing.

**Why it happens:**
New code entangled with existing code. Shared state modified incorrectly. Timing dependencies not understood.

**Consequences:**
Working Tetris game becomes broken. Regression of core functionality. All players affected.

**Prevention:**
Keep new features isolated. Use feature flags for gradual rollout. Extensive regression testing of core mechanics after each addition. Do not modify existing rotation, collision, or timing code.

**Detection:**
Core gameplay broken after adding new feature. Pieces behave differently. Timing off.

### Critical Pitfall 30: Race Conditions Between New Systems

**What goes wrong:**
Combo system calculates score before T-spin detection runs, missing T-spin bonus. Or audio plays for wrong action due to timing.

**Why it happens:**
Multiple systems run in same game loop without defined order. Async operations complete out of order.

**Consequences:**
Inconsistent scoring. Audio mismatch. Difficult-to-reproduce bugs.

**Prevention:**
Define explicit execution order for game loop phases: input → update → collision → scoring → audio → render. Document dependencies between systems.

**Detection:**
Scoring inconsistent. Sometimes T-spin bonus applies, sometimes does not. Timing-dependent bugs.

### Moderate Pitfall 31: Feature Flag Explosion

**What goes wrong:**
Every new feature gets feature flag, resulting in combinatorial explosion of test cases and code paths.

**Consequences:**
Code becomes unmaintainable. Impossible to test all flag combinations.

**Prevention:**
Use feature flags for rollout only, remove after stable. Maximum 2-3 active flags at once. Progressive enhancement instead of flags when possible.

**Detection:**
Code filled with if (featureFlag) checks. Test matrix exploding.

## Prevention Strategies by Phase

| Phase Topic | Recommended Approach | When to Validate |
|-------------|---------------------|------------------|
| T-Spin Detection | Implement detection as read-only check separate from rotation. Test all boundary conditions first. | After rotation system verified working |
| Combo System | Unit test lookup tables extensively. Store previous level for multiplier calculation. | Before integration with scoring |
| Audio System | Object pool for sound effects. Detect autoplay restrictions early with user gesture. | First user interaction test |
| Key Remapping | Validate conflicts before save. Use KeyboardEvent.code. Persist to localStorage. | After default controls confirmed working |
| Statistics | Use localStorage not sessionStorage. Implement quota error handling. | First stats save attempt |
| Theme System | CSS variables not separate stylesheets. Inline script in head for preference. | Page load |
| Integration | Feature isolation. Defined system execution order. Extensive regression testing. | After each feature addition |

## Testing Checklist

Before marking any phase complete:

- [ ] Core mechanics still work (rotation, collision, line clear)
- [ ] T-spins detected at walls, floor, and center field
- [ ] Audio plays after user interaction, no memory leaks
- [ ] Key remapping persists across sessions, no conflicts
- [ ] Statistics saved to localStorage, quota errors handled
- [ ] Theme switch has no flash, respects system preference
- [ ] No race conditions between systems
- [ ] Performance acceptable (60fps maintained)
- [ ] Mobile devices tested (audio, touch controls)

## Confidence Assessment

| Area | Confidence | Reason |
|------|-----------|--------|
| T-Spin Pitfalls | MEDIUM | Multiple sources from Tetris community wikis, documented bugs from commercial games |
| Combo Pitfalls | MEDIUM | Documented bugs from TGM series and Puyo Puyo Tetris |
| Audio Pitfalls | HIGH | Official MDN documentation, Howler.js GitHub issues, Web Audio API spec |
| Key Remapping | MEDIUM | Web development best practices, browser API documentation |
| Statistics | MEDIUM | Browser storage API documentation, established patterns |
| Theme System | MEDIUM | CSS best practices, documented FOUC solutions |
| Integration | HIGH | Software engineering best practices, common patterns |

## Sources

**T-Spin Detection:**
- [T-Spin - TetrisWiki](https://tetris.wiki/T-Spin)
- [T-Spin - Hard Drop Tetris Wiki](https://harddrop.com/wiki/T-Spin)
- [T-Spin Guide - Hard Drop Tetris Wiki](https://harddrop.com/wiki/T-Spin_Guide)
- [Tetris Aside: Coding for T-Spins | Katy's Code](https://katyscode.wordpress.com/2012/10/13/tetris-aside-coding-for-t-spins/)
- [The Tetra-X kickset incorrectly defines at least one T-Spin Mini scenario](https://github.com/tetrio/issues/issues/626)

**Combo System:**
- [Combo - TetrisWiki](https://tetris.wiki/Combo)
- [Combo - Hard Drop Tetris Wiki](https://harddrop.com/wiki/Combo)
- [Scoring - TetrisWiki](https://tetris.wiki/Scoring)
- [List of TGM series bugs - TetrisWiki](https://tetris.wiki/List_of_TGM_series_bugs)

**Audio:**
- [Autoplay guide for media and Web Audio APIs - MDN](https://developer.mozilla.org/en-US/docs/Web/Media/Autoplay_guide)
- [Autoplay policy in Chrome | Blog](https://developer.chrome.com/blog/autoplay)
- [Memory leak on unload() method - Howler.js Issue #914](https://github.com/goldfire/howler.js/issues/914)
- [Unload not releasing memory - Howler.js Issue #1731](https://github.com/goldfire/howler.js/issues/1731)
- [Memory Leak on streaming - Howler.js Discussion #1580](https://github.com/goldfire/howler.js/discussions/1580)
- [5 Audio Pitfalls Every Game Developer Should Know](https://www.thegameaudioco.com/5-audio-pitfalls-every-game-developer-should-know)
- [Optimizing Game Audio for Peak Performance](https://www.numberanalytics.com/blog/optimizing-game-audio-for-peak-performance)

**Storage & Statistics:**
- [HTML LocalStorage and SessionStorage - GeeksforGeeks](https://www.geeksforgeeks.org/javascript/localstorage-and-sessionstorage-web-storage-apis/)
- [Best Practices for Persisting State in Frontend Applications](https://blog.pixelfreestudio.com/best-practices-for-persisting-state-in-frontend-applications/)
- [LocalStorage, sessionStorage - JavaScript.info](https://javascript.info/localstorage)

**Key Remapping:**
- [Glossary:Remapping - PCGamingWiki](https://www.pcgamingwiki.com/wiki/Glossary:Remapping)
- [Remap Keys and Shortcuts with PowerToys Keyboard Manager](https://learn.microsoft.com/en-us/windows/powertoys/keyboard-manager)

**Theme System:**
- [A (mostly complete) guide to theme switching in CSS and JS](https://medium.com/@cerutti.alexander/a-mostly-complete-guide-to-theme-switching-in-css-and-js-c4992d5fd357)
- [The Perfect Theme Switch Component](https://www.aleksandrhovhannisyan.com/blog/the-perfect-theme-switch/)
- [Building an accessible theme picker](https://fossheim.io/writing/posts/accessible-theme-picker-html-css-js/)

**SRS & Rotation:**
- [Super Rotation System - TetrisWiki](https://tetris.wiki/Super_Rotation_System)
- [SRS - Hard Drop Tetris Wiki](https://harddrop.com/wiki/SRS)
- [Wall kick - TetrisWiki](https://tetris.wiki/Wall_kick)

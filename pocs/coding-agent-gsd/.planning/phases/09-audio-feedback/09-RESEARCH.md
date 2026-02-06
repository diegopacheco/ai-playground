# Phase 9: Audio Feedback - Research

**Researched:** 2026-02-05
**Domain:** Web Audio API / Browser Audio
**Confidence:** HIGH

## Summary

Web audio implementation for games has two primary approaches: HTML5 `<audio>` element and Web Audio API. For this phase, vanilla JavaScript using Web Audio API is the standard approach for game sound effects, offering precise timing control essential for responsive game feedback. The technical constraint of no external dependencies aligns perfectly with the native Web Audio API.

The Web Audio API has been baseline widely available across all modern browsers since April 2021, making it a safe choice. The API uses an audio graph architecture where source nodes connect to destination (speakers). A critical design pattern is that AudioBufferSourceNode instances are single-use but inexpensive to create, while AudioBuffer objects containing the actual audio data are expensive and should be reused.

For the requirements in this phase, the recommended approach is:
1. Use Web Audio API OscillatorNode to generate simple sound effects programmatically
2. Implement AudioBuffer pooling pattern for pre-loaded sounds if needed
3. Synchronize mute state between admin panel and game using existing BroadcastChannel API infrastructure
4. Persist mute preference to localStorage with proper initialization checking

**Primary recommendation:** Use Web Audio API with OscillatorNode-generated sounds for all game events, avoiding external audio files entirely to maintain zero-dependency constraint.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Web Audio API | Native | Audio generation and playback | Built into all modern browsers since 2021, zero dependencies |
| localStorage API | Native | Persist user preferences | Standard browser storage for settings |
| BroadcastChannel API | Native | Cross-tab synchronization | Already in use per STATE.md decisions |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| AudioContext | Native | Central audio processing graph | Required for all Web Audio operations |
| OscillatorNode | Native | Generate sound waves | Simple beeps, tones without audio files |
| AudioBufferSourceNode | Native | Play pre-loaded audio | If using actual audio file assets |
| GainNode | Native | Volume control | Implement mute/volume features |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Web Audio API | HTML5 `<audio>` element | Simpler API but lacks precise timing control needed for game events |
| Generated sounds | Audio file sprites | Better sound quality but requires external dependencies (audio files) |
| Web Audio API | Third-party libraries (howler.js, tone.js) | Easier API but violates TECH-04 no external dependencies constraint |

**Installation:**
None required - all APIs are native to browsers.

## Architecture Patterns

### Recommended Project Structure
```
js/
├── audio/
│   ├── audio-manager.js     # AudioContext initialization, sound playback API
│   ├── sound-effects.js     # OscillatorNode-based sound generation functions
│   └── audio-settings.js    # Mute state management and persistence
```

### Pattern 1: AudioContext Singleton
**What:** Single AudioContext instance shared across application
**When to use:** Always - multiple contexts cause performance degradation

**Example:**
```javascript
let audioContext = null;

function getAudioContext() {
  if (!audioContext) {
    audioContext = new AudioContext();
    if (audioContext.state === 'suspended') {
      audioContext.resume();
    }
  }
  return audioContext;
}
```

### Pattern 2: AudioBuffer Reuse with Source Node Pool
**What:** Create one AudioBuffer, generate new AudioBufferSourceNode for each playback
**When to use:** When using pre-loaded audio files (not needed if generating sounds)

**Example:**
```javascript
const audioBuffer = await loadSound('sound.mp3');

function playSound() {
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);
  source.start();
}
```

### Pattern 3: Programmatic Sound Generation
**What:** Use OscillatorNode to generate sound effects without audio files
**When to use:** Simple game sounds (beeps, blips) and zero-dependency requirement

**Example:**
```javascript
function playBeep(frequency = 440, duration = 0.1) {
  const ctx = getAudioContext();
  const oscillator = ctx.createOscillator();
  const gainNode = ctx.createGain();

  oscillator.connect(gainNode);
  gainNode.connect(ctx.destination);

  oscillator.frequency.value = frequency;
  oscillator.type = 'sine';

  gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration);

  oscillator.start(ctx.currentTime);
  oscillator.stop(ctx.currentTime + duration);
}
```

### Pattern 4: Mute State Synchronization
**What:** Use BroadcastChannel to sync mute setting between admin and game
**When to use:** Required for AUDIO-05 and existing BroadcastChannel infrastructure

**Example:**
```javascript
const audioChannel = new BroadcastChannel('game_audio');

function setMuted(muted) {
  localStorage.setItem('audio_muted', JSON.stringify(muted));
  audioChannel.postMessage({ action: 'mute_changed', muted });
}

audioChannel.addEventListener('message', (event) => {
  if (event.data.action === 'mute_changed') {
    applyMuteState(event.data.muted);
  }
});

function initMuteState() {
  const stored = localStorage.getItem('audio_muted');
  return stored !== null ? JSON.parse(stored) : false;
}
```

### Pattern 5: Autoplay Policy Compliance
**What:** Resume AudioContext on user interaction
**When to use:** Always - modern browsers require user gesture before audio

**Example:**
```javascript
document.addEventListener('click', () => {
  const ctx = getAudioContext();
  if (ctx.state === 'suspended') {
    ctx.resume();
  }
}, { once: true });
```

### Anti-Patterns to Avoid

**Reusing AudioBufferSourceNode:** Once `start()` is called, the node cannot be restarted. Create new nodes for each playback.

**Multiple AudioContext instances:** Creates performance degradation. Use singleton pattern.

**Ignoring autoplay policy:** Always check and resume AudioContext state on user interaction.

**Storing paused state in variable:** Read `audio.paused` property live, do not cache it.

**Hard stops without ramping:** Causes clicking sounds. Use `exponentialRampToValueAtTime()` to fade out.

**Setting gain to zero:** Use very small value (0.01) instead of zero for exponential ramps, as exponential functions cannot reach zero.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cross-tab settings sync | Custom localStorage polling | BroadcastChannel API | Real-time, event-driven, no polling overhead |
| Audio context management | Manual state tracking | Check `audioContext.state` property | Browser manages state, policies change |
| Smooth audio stop | Immediate `stop()` call | `GainNode` with `exponentialRampToValueAtTime()` | Prevents clicking artifacts |
| Mute implementation | Disconnecting nodes | `GainNode.gain.value = 0` | Cleaner, reversible, standard pattern |
| Sound effect timing | setTimeout/setInterval | `AudioContext.currentTime` + scheduled starts | Sub-millisecond precision, won't drift |

**Key insight:** Web Audio API provides high-precision timing and audio graph architecture. Trying to manage timing externally with JavaScript timers introduces drift and imprecision.

## Common Pitfalls

### Pitfall 1: AudioContext Suspended by Autoplay Policy
**What goes wrong:** Creating AudioContext before user interaction results in `suspended` state, no audio plays.
**Why it happens:** Modern browsers require user gesture before allowing audio playback to prevent annoying auto-play ads.
**How to avoid:** Check `audioContext.state` and resume on first user interaction.
**Warning signs:** Audio code runs but no sound plays, console shows no errors.

### Pitfall 2: Reusing Source Nodes
**What goes wrong:** Calling `start()` on same AudioBufferSourceNode or OscillatorNode twice throws error.
**Why it happens:** Source nodes are designed for single use by Web Audio API specification.
**How to avoid:** Create new source node for each playback, reuse only the AudioBuffer data.
**Warning signs:** Error: "InvalidStateError: Failed to execute 'start' on AudioBufferSourceNode."

### Pitfall 3: Clicking Sounds on Audio Stop
**What goes wrong:** Abruptly stopping audio creates audible click or pop.
**Why it happens:** Stopping audio mid-waveform creates discontinuity in the audio signal.
**How to avoid:** Ramp gain down with `exponentialRampToValueAtTime()` before stopping.
**Warning signs:** Unwanted clicking/popping sounds when effects end.

### Pitfall 4: localStorage Overwriting on Page Load
**What goes wrong:** Mute preference resets to default on every page load.
**Why it happens:** Initialization code sets default value unconditionally, overwriting persisted data.
**How to avoid:** Check `if (!localStorage.getItem(key))` before setting default.
**Warning signs:** User toggles mute, but setting doesn't persist across refreshes.

### Pitfall 5: Exponential Ramp to Zero
**What goes wrong:** Using `exponentialRampToValueAtTime(0, time)` throws error.
**Why it happens:** Exponential functions mathematically cannot reach zero.
**How to avoid:** Use very small value like 0.01 instead of zero, or use `linearRampToValueAtTime()`.
**Warning signs:** Error: "RangeError: The value provided is outside the range."

### Pitfall 6: Memory Leaks from Unclosed AudioContext
**What goes wrong:** Creating and destroying AudioContext repeatedly causes memory accumulation.
**Why it happens:** AudioContext holds onto system resources until explicitly closed.
**How to avoid:** Use singleton pattern, call `audioContext.close()` only when truly done with audio.
**Warning signs:** Increasing memory usage over time on mobile devices, battery drain.

## Code Examples

Verified patterns from official sources:

### Basic Beep Sound (OscillatorNode)
```javascript
function playBeep(frequency, duration) {
  const ctx = new AudioContext();
  const oscillator = ctx.createOscillator();
  const gainNode = ctx.createGain();

  oscillator.connect(gainNode);
  gainNode.connect(ctx.destination);

  oscillator.frequency.value = frequency;
  oscillator.type = 'sine';

  gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration);

  oscillator.start(ctx.currentTime);
  oscillator.stop(ctx.currentTime + duration);
}

playBeep(440, 0.1);
playBeep(880, 0.15);
```

### Audio Sprite Pattern (if using files)
```javascript
const myAudio = document.createElement('audio');
myAudio.src = 'sprite.mp3';

const sounds = {
  land: { start: 0, stop: 0.1 },
  clear: { start: 0.6, stop: 0.8 },
  tetris: { start: 1.2, stop: 1.5 },
  gameover: { start: 2.0, stop: 2.5 }
};

function playSound(name) {
  const sound = sounds[name];
  myAudio.currentTime = sound.start;
  myAudio.play();

  const checkTime = () => {
    if (myAudio.currentTime >= sound.stop) {
      myAudio.pause();
    } else {
      requestAnimationFrame(checkTime);
    }
  };
  requestAnimationFrame(checkTime);
}
```

### Mute Toggle with Persistence
```javascript
const channel = new BroadcastChannel('game_audio');
let isMuted = false;

function initAudio() {
  const stored = localStorage.getItem('audio_muted');
  if (stored !== null) {
    isMuted = JSON.parse(stored);
  }
  return isMuted;
}

function toggleMute() {
  isMuted = !isMuted;
  localStorage.setItem('audio_muted', JSON.stringify(isMuted));
  channel.postMessage({ action: 'mute_changed', muted: isMuted });
}

channel.addEventListener('message', (event) => {
  if (event.data.action === 'mute_changed') {
    isMuted = event.data.muted;
    updateMuteUI(isMuted);
  }
});

isMuted = initAudio();
```

### AudioContext with Autoplay Handling
```javascript
let audioContext = null;

function getAudioContext() {
  if (!audioContext) {
    audioContext = new AudioContext();
  }

  if (audioContext.state === 'suspended') {
    audioContext.resume();
  }

  return audioContext;
}

document.addEventListener('click', () => {
  getAudioContext();
}, { once: true });
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ScriptProcessorNode | AudioWorklet | 2017-2018 | Better performance, runs off main thread |
| Prefixed AudioContext | Standard AudioContext | 2014 | webkitAudioContext deprecated, use AudioContext |
| Multiple audio files | Audio sprites or generated sounds | Ongoing | Fewer HTTP requests, faster load times |
| localStorage events for sync | BroadcastChannel API | 2016-2021 | Real-time, no disk I/O overhead |
| Manual autoplay | Autoplay policy compliance | 2018 | Must resume context on user gesture |

**Deprecated/outdated:**
- **ScriptProcessorNode:** Replaced by AudioWorklet for custom audio processing.
- **webkitAudioContext:** Use unprefixed `AudioContext` - supported since 2014 in all browsers.
- **createJavaScriptNode:** Old Firefox method, use createScriptProcessor (or better, AudioWorklet).

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal frequencies for game events**
   - What we know: Lower frequencies (200-400 Hz) sound like thuds, higher (800-1200 Hz) sound like blips.
   - What's unclear: Exact frequencies that feel satisfying for Tetris events.
   - Recommendation: Start with 220 Hz for piece land, 440 Hz for line clear, 880 Hz for Tetris, 110 Hz for game over. Tune by ear during implementation.

2. **Waveform types for different events**
   - What we know: Sine waves are pure tones, square waves are harsher, triangle are softer, sawtooth are bright.
   - What's unclear: Which waveform feels best for each game event.
   - Recommendation: Default to sine for clean sounds, experiment with triangle for softer landing sound.

3. **Sound duration sweet spot**
   - What we know: Too short (< 50ms) sounds like click, too long (> 300ms) feels sluggish.
   - What's unclear: Exact duration that feels responsive but not intrusive.
   - Recommendation: Start with 100ms for land/clear, 200ms for Tetris, 500ms for game over. Test and adjust.

## Sources

### Primary (HIGH confidence)
- [MDN: Audio for Web Games](https://developer.mozilla.org/en-US/docs/Games/Techniques/Audio_for_Web_Games) - Game audio implementation guide
- [MDN: Web Audio API Best Practices](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices) - Official best practices
- [MDN: AudioBufferSourceNode](https://developer.mozilla.org/en-US/docs/Web/API/AudioBufferSourceNode) - Source node reuse patterns
- [MDN: Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) - API reference
- [Chrome Developers: HTML5 Audio and Web Audio API](https://developer.chrome.com/blog/html5-audio-and-the-web-audio-api-are-bffs) - Combining approaches
- [MDN Blog: BroadcastChannel API for Cross-Tab Communication](https://developer.mozilla.org/en-US/blog/exploring-the-broadcast-channel-api-for-cross-tab-communication/) - Tab sync patterns

### Secondary (MEDIUM confidence)
- [web.dev: Audio Effects Patterns](https://web.dev/patterns/media/audio-effects) - Effect implementation
- [Medium: Syncing Data Across Tabs with BroadcastChannel](https://medium.com/@sachin88/syncing-data-across-browser-tabs-with-the-broadcastchannel-api-de26f61529fb) - Settings sync
- [DEV: Creating an Oscillator](https://dev.to/rayalva407/creating-an-oscillator-with-the-web-audio-api-5b8m) - Sound generation tutorial

### Tertiary (LOW confidence)
- [CodePen: Javascript Beep](https://codepen.io/noraspice/pen/JpVXVP) - Simple beep example
- [GitHub: browser-beep](https://github.com/kapetan/browser-beep) - Beep implementation reference
- [marcgg.com: Generate Sounds Programmatically](https://marcgg.com/blog/2016/11/01/javascript-audio/) - Sound generation guide

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All APIs are native browser features with MDN documentation
- Architecture: HIGH - Patterns verified from official MDN guides and Chrome developers blog
- Pitfalls: HIGH - Common issues documented in MDN best practices and developer discussions
- Code examples: HIGH - Sourced from official MDN documentation and web.dev
- Sound design (frequencies/durations): LOW - Requires testing and iteration during implementation

**Research date:** 2026-02-05
**Valid until:** 2026-03-05 (30 days - stable APIs, but autoplay policies can change)

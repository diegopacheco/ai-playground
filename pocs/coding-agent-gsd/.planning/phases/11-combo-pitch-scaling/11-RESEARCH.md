# Phase 11: Combo Pitch Scaling - Research

**Researched:** 2026-02-06
**Domain:** Web Audio API dynamic pitch scaling for game audio feedback
**Confidence:** HIGH

## Summary

Combo pitch scaling in Tetris Twist requires dynamically adjusting the frequency of line clear sound effects based on the current combo counter. The research confirms this is achievable with the existing Web Audio API OscillatorNode architecture with zero new dependencies. The critical technical requirement is using exponentialRampToValueAtTime for parameter automation to prevent audio clicks during frequency transitions.

The existing audio.js implementation uses OscillatorNode synthesis with basic playSound function that creates throwaway oscillator nodes for each effect. The combo counter is already tracked in main.js and incremented on line clears. Integration requires passing the combo value to playLineClearSound and applying frequency scaling with the formula: baseFrequency * (1 + scalingFactor * Math.min(combo, cap)).

**Primary recommendation:** Use exponentialRampToValueAtTime with 30ms ramp duration for all frequency changes, cap combo multiplier at 10x to prevent painful high frequencies, and implement different base frequencies for single/double/triple/Tetris clears. Safari compatibility requires no fallback since OscillatorNode.frequency is universally supported, unlike the detune property.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Web Audio API | Native | Audio synthesis and parameter automation | Universal browser support, zero dependencies, precise timing control |
| OscillatorNode | Native | Frequency-based sound generation | Already implemented in audio.js, supports AudioParam automation |
| AudioParam | Native | Parameter ramping and scheduling | Prevents clicks through exponentialRampToValueAtTime |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| AudioContext.sampleRate | Native | Nyquist limit calculation | Cap maximum frequency at sampleRate/2 * 0.9 to prevent aliasing |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| OscillatorNode frequency | AudioBufferSourceNode playbackRate | BufferSource requires audio files and is one-time use, increasing complexity |
| exponentialRampToValueAtTime | linearRampToValueAtTime | Linear ramps sound unnatural for pitch changes due to logarithmic human hearing |
| exponentialRampToValueAtTime | Direct frequency.value assignment | Direct assignment causes audible clicks from waveform discontinuities |
| Web Audio API | Howler.js or Tone.js | 50-100KB overhead for capabilities already present in codebase |

**Installation:**
Zero new dependencies required. Feature uses existing Web Audio API implementation.

## Architecture Patterns

### Recommended Project Structure
Existing structure is already correct:
```
js/
├── audio.js         # Extend playLineClearSound to accept combo parameter
├── main.js          # Pass combo value when calling playLineClearSound
└── render.js        # No changes needed, combo already displayed
```

### Pattern 1: Frequency Scaling with Combo Multiplier
**What:** Calculate target frequency based on base frequency and combo counter with capping
**When to use:** Every line clear sound effect that should reward combo chains
**Example:**
```javascript
function playLineClearSound(combo) {
    const baseFrequency = 440;
    const scalingFactor = 0.1;
    const maxCombo = 10;
    const clampedCombo = Math.min(combo, maxCombo);
    const targetFrequency = baseFrequency * (1 + scalingFactor * clampedCombo);
    playSound(targetFrequency, 100);
}
```

### Pattern 2: Parameter Ramping for Click Prevention
**What:** Use exponentialRampToValueAtTime instead of direct value assignment
**When to use:** All dynamic frequency changes to prevent audible artifacts
**Example:**
```javascript
function playSound(frequency, duration) {
    if (muted) return;
    const ctx = getAudioContext();
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();

    oscillator.connect(gain);
    gain.connect(ctx.destination);

    oscillator.frequency.setValueAtTime(frequency, ctx.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(
        frequency,
        ctx.currentTime + 0.03
    );
    oscillator.type = 'sine';

    gain.gain.value = 0.1;
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration / 1000);

    oscillator.start(ctx.currentTime);
    oscillator.stop(ctx.currentTime + duration / 1000);
}
```

### Pattern 3: Differentiated Base Frequencies
**What:** Use different starting frequencies for single/double/triple/Tetris clears
**When to use:** Provide audio variety that matches clear difficulty
**Example:**
```javascript
function playLineClearSound(linesCleared, combo) {
    const baseFrequencies = {
        1: 330,
        2: 440,
        3: 550,
        4: 660
    };
    const baseFrequency = baseFrequencies[linesCleared] || 440;
    const scalingFactor = 0.1;
    const maxCombo = 10;
    const clampedCombo = Math.min(combo, maxCombo);
    const targetFrequency = baseFrequency * (1 + scalingFactor * clampedCombo);
    playSound(targetFrequency, 100);
}
```

### Pattern 4: Nyquist Limit Safety
**What:** Cap maximum frequency at 90% of Nyquist frequency to prevent aliasing
**When to use:** Always, as final validation before playing any frequency
**Example:**
```javascript
function playSound(frequency, duration) {
    if (muted) return;
    const ctx = getAudioContext();
    const nyquistLimit = ctx.sampleRate / 2 * 0.9;
    const safeFrequency = Math.min(frequency, nyquistLimit);

    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();

    oscillator.connect(gain);
    gain.connect(ctx.destination);

    oscillator.frequency.setValueAtTime(safeFrequency, ctx.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(
        safeFrequency,
        ctx.currentTime + 0.03
    );
    oscillator.type = 'sine';

    gain.gain.value = 0.1;
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration / 1000);

    oscillator.start(ctx.currentTime);
    oscillator.stop(ctx.currentTime + duration / 1000);
}
```

### Anti-Patterns to Avoid
- **Direct frequency.value assignment:** Causes audible clicks from waveform discontinuities. Always use setValueAtTime + exponentialRampToValueAtTime.
- **Linear ramping for pitch:** Sounds unnatural due to logarithmic human hearing. Use exponentialRampToValueAtTime.
- **Unbounded combo scaling:** High combos produce ear-piercing frequencies. Cap at 10x multiplier.
- **Forgetting Nyquist limit:** Frequencies above sampleRate/2 cause aliasing artifacts. Always validate against 90% of Nyquist.
- **Using detune property:** Safari doesn't support detune. Use frequency directly for universal compatibility.
- **Zero values in exponential ramps:** exponentialRampToValueAtTime throws error for values <= 0. Use 0.01 as minimum.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Frequency to pitch conversion | Custom semitone calculator | Direct frequency multiplication with musical ratios | Equal temperament formula (2^(n/12)) is standard but unnecessary for linear scaling in this use case |
| Parameter ramping | Custom setTimeout fade loops | AudioParam.exponentialRampToValueAtTime | Native method is sample-accurate, handles timing internally, prevents clicks |
| Click prevention | Custom fade-in/fade-out with multiple oscillators | AudioParam automation methods | Web Audio API provides precise parameter scheduling that prevents discontinuities |
| Safari compatibility detection | Browser sniffing for detune support | Use frequency property directly | OscillatorNode.frequency is universally supported, detune is optional enhancement |

**Key insight:** Web Audio API's AudioParam automation handles the hard problems of sample-accurate timing, click prevention, and smooth parameter transitions. Custom implementations introduce timing errors, audible artifacts, and browser inconsistencies.

## Common Pitfalls

### Pitfall 1: Direct Frequency Assignment Causes Clicks
**What goes wrong:** Setting frequency.value directly causes audible clicks and pops
**Why it happens:** Instantaneous frequency changes create waveform discontinuities that produce transient artifacts
**How to avoid:** Always use setValueAtTime followed by exponentialRampToValueAtTime with 30ms duration
**Warning signs:** Hearing clicks when combo increases, especially noticeable on headphones

### Pitfall 2: Zero or Negative Values in Exponential Ramps
**What goes wrong:** exponentialRampToValueAtTime throws DOMException for values <= 0
**Why it happens:** Exponential curves mathematically cannot reach zero, API enforces this constraint
**How to avoid:** Use 0.01 as minimum value for gain fade-outs, never 0
**Warning signs:** Console errors with "InvalidAccessError" when ramping gain down

### Pitfall 3: Unbounded Combo Scaling
**What goes wrong:** High combos produce ear-piercing frequencies above 2000Hz that hurt
**Why it happens:** Linear scaling without cap multiplies base frequency indefinitely
**How to avoid:** Cap combo multiplier with Math.min(combo, 10) before frequency calculation
**Warning signs:** Sound becomes painful at high combos, players report discomfort

### Pitfall 4: Exceeding Nyquist Frequency
**What goes wrong:** Frequencies above sampleRate/2 cause aliasing artifacts and distortion
**Why it happens:** Digital audio can't represent frequencies above half the sample rate
**How to avoid:** Clamp final frequency to ctx.sampleRate / 2 * 0.9 before playing
**Warning signs:** Distorted or garbled sound at high frequencies, unexpected harmonics

### Pitfall 5: Confusing Linear vs Exponential Ramping
**What goes wrong:** Using linearRampToValueAtTime for pitch changes sounds unnatural
**Why it happens:** Human hearing perceives pitch logarithmically, not linearly
**How to avoid:** Always use exponentialRampToValueAtTime for frequency changes
**Warning signs:** Pitch transitions sound mechanical or artificial

### Pitfall 6: Forgetting to Pass Combo to Sound Function
**What goes wrong:** Line clear sound never changes pitch despite combo system working
**Why it happens:** main.js calls playLineClearSound() without combo parameter
**How to avoid:** Modify call site to playLineClearSound(combo) and update function signature
**Warning signs:** Combo counter displays correctly but audio stays constant

### Pitfall 7: Ramp Duration Too Short or Too Long
**What goes wrong:** Too short (< 10ms) doesn't prevent clicks, too long (> 50ms) causes lag
**Why it happens:** Click prevention needs time to ramp but excessive duration delays audio feedback
**How to avoid:** Use 30ms (0.03s) as sweet spot for imperceptible but effective ramping
**Warning signs:** Still hearing clicks (too short) or delayed audio response (too long)

### Pitfall 8: Reusing OscillatorNode Instances
**What goes wrong:** Attempting to start() an oscillator that was already stopped throws error
**Why it happens:** OscillatorNodes are one-time use, cannot be restarted after stop()
**How to avoid:** Create new oscillator for each playSound call, existing implementation already correct
**Warning signs:** Console errors "InvalidStateError: Failed to execute 'start' on 'OscillatorNode'"

## Code Examples

Verified patterns from official sources:

### Basic Frequency Scaling with Combo
```javascript
function playLineClearSound(combo) {
    const baseFrequency = 440;
    const scalingFactor = 0.1;
    const maxCombo = 10;
    const clampedCombo = Math.min(combo || 0, maxCombo);
    const targetFrequency = baseFrequency * (1 + scalingFactor * clampedCombo);
    playSound(targetFrequency, 100);
}
```

### Parameter Ramping for Click Prevention
```javascript
function playSound(frequency, duration) {
    if (muted) return;
    const ctx = getAudioContext();
    const nyquistLimit = ctx.sampleRate / 2 * 0.9;
    const safeFrequency = Math.min(frequency, nyquistLimit);

    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();

    oscillator.connect(gain);
    gain.connect(ctx.destination);

    oscillator.frequency.setValueAtTime(safeFrequency, ctx.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(
        safeFrequency,
        ctx.currentTime + 0.03
    );
    oscillator.type = 'sine';

    gain.gain.value = 0.1;
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration / 1000);

    oscillator.start(ctx.currentTime);
    oscillator.stop(ctx.currentTime + duration / 1000);
}
```

### Different Frequencies per Clear Type
```javascript
function getBaseFrequency(linesCleared) {
    const frequencies = {
        1: 330,
        2: 440,
        3: 550,
        4: 660
    };
    return frequencies[linesCleared] || 440;
}

function playLineClearSound(linesCleared, combo) {
    const baseFrequency = getBaseFrequency(linesCleared);
    const scalingFactor = 0.1;
    const maxCombo = 10;
    const clampedCombo = Math.min(combo || 0, maxCombo);
    const targetFrequency = baseFrequency * (1 + scalingFactor * clampedCombo);
    playSound(targetFrequency, 100);
}
```

### Integration Point in main.js
```javascript
if (result.linesCleared === 4) {
    playTetrisSound();
} else if (result.linesCleared > 0) {
    playLineClearSound(result.linesCleared, combo);
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Static sound frequencies | Dynamic pitch based on combo | 2015+ (Tetris Effect) | Creates reward feedback loop that increases player engagement |
| Direct frequency.value = X | setValueAtTime + exponentialRampToValueAtTime | 2013 (Web Audio spec) | Eliminates clicks and artifacts from parameter changes |
| Audio files for variants | OscillatorNode frequency scaling | 2013+ | Zero file loading, infinite pitch variations, smaller footprint |
| detune property for pitch | frequency property directly | 2020+ (Safari compat) | Universal browser support without fallback code |

**Deprecated/outdated:**
- Using detune for OscillatorNode pitch shifting: Safari never implemented it, frequency property is more widely supported
- linearRampToValueAtTime for pitch: Exponential is standard for frequency changes due to human hearing perception
- Unbounded pitch scaling: Modern games cap at 2x-3x multiplier to prevent discomfort

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal scaling factor for combo pitch**
   - What we know: Linear scaling with 10% per combo (scalingFactor = 0.1) suggested in v3.0 research
   - What's unclear: Whether 10% steps sound good or need tuning
   - Recommendation: Start with 0.1, playtest and adjust. Range 0.05-0.15 likely optimal.

2. **Should Tetris clear use separate scaling**
   - What we know: Tetris already has different sound (playTetrisSound at 880Hz)
   - What's unclear: Should Tetris clear also scale with combo or stay constant
   - Recommendation: Apply combo scaling to Tetris too for consistency, but test both approaches

3. **Exact ramp duration for best click prevention**
   - What we know: Research suggests 15ms (setTargetAtTime) or 30ms (exponentialRampToValueAtTime)
   - What's unclear: Whether Tetris needs 30ms or could use faster 15ms
   - Recommendation: Start with 30ms as safer option, reduce to 15ms if audio feedback feels sluggish

## Sources

### Primary (HIGH confidence)
- [MDN Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) - Official API documentation
- [MDN OscillatorNode.frequency](https://developer.mozilla.org/en-US/docs/Web/API/OscillatorNode/frequency) - Frequency property specification
- [MDN AudioParam](https://developer.mozilla.org/en-US/docs/Web/API/AudioParam) - Parameter automation methods
- [MDN exponentialRampToValueAtTime](https://developer.mozilla.org/en-US/docs/Web/API/AudioParam/exponentialRampToValueAtTime) - Ramping specification
- [Web Audio: the ugly click and the human ear](http://alemangui.github.io/ramp-to-value) - Parameter ramping best practices with exact timing values

### Secondary (MEDIUM confidence)
- [Pitch shifting in Web Audio API](https://zpl.fi/pitch-shifting-in-web-audio-api/) - Safari detune issue and playbackRate fallback
- [Game audio analysis - Tetris Effect](https://www.gamedeveloper.com/audio/game-audio-analysis---tetris-effect) - Modern Tetris audio design patterns
- [Nyquist Frequency Explained](https://mixingmonster.com/nyquist-frequency-explained/) - Aliasing prevention and sample rate limits
- [Piano key frequencies](https://en.wikipedia.org/wiki/Piano_key_frequencies) - Musical frequency formulas and A4=440Hz standard
- [The Power of Pitch Shifting](https://www.gamedeveloper.com/audio/the-power-of-pitch-shifting) - Game audio pitch scaling patterns

### Tertiary (LOW confidence)
- [Web Audio API – things I learned the hard way](https://blog.szynalski.com/2014/04/web-audio-api/) - OscillatorNode throwaway pattern
- Tetris Twist v3.0 SUMMARY.md - Prior research with combo pitch formula suggestion

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Web Audio API is proven in existing codebase, zero new dependencies needed
- Architecture: HIGH - Clear integration points identified, existing combo tracking ready to use
- Pitfalls: HIGH - Click prevention patterns verified from multiple authoritative sources with exact timing values

**Research date:** 2026-02-06
**Valid until:** 2026-03-06 (30 days - stable API patterns)

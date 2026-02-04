# Pitfalls Research: Browser Tetris Game

## Critical Pitfalls

### 1. Game Loop Timing Issues

**Problem:** Using `setInterval` for game loop causes drift, inconsistent frame rates, and freezes during tab switches.

**Warning Signs:**
- Pieces fall at inconsistent speeds
- Game speeds up after tab is backgrounded
- Jerky movement

**Prevention:**
- Use `requestAnimationFrame` for rendering
- Track delta time between frames
- Pause game loop when tab is hidden (`visibilitychange` event)

**Phase:** Core game loop (Phase 1)

---

### 2. Rotation Near Walls (Wall Kicks)

**Problem:** Pieces can't rotate near edges, frustrating players. Or worse — pieces rotate into walls/other pieces.

**Warning Signs:**
- Rotation fails silently near edges
- Pieces overlap after rotation
- Players complain rotation "doesn't work"

**Prevention:**
- Implement wall kick offsets (try alternative positions when rotation blocked)
- Test rotation at all board edges
- At minimum: check collision before applying rotation

**Phase:** Piece mechanics (Phase 2)

---

### 3. Canvas Scaling/Blur

**Problem:** Canvas renders blurry on high-DPI displays (Retina). Grid lines look fuzzy.

**Warning Signs:**
- Blurry rendering on MacBooks, modern phones
- Grid doesn't align with pixels

**Prevention:**
- Scale canvas by `devicePixelRatio`
- Set canvas CSS size separate from internal resolution
- Use integer coordinates for grid lines

**Phase:** Rendering setup (Phase 1)

---

### 4. Growing Board Breaks Layout

**Problem:** When board grows, canvas overflows container, pieces get displaced, or game becomes unplayable.

**Warning Signs:**
- Canvas extends beyond viewport
- Pieces appear in wrong positions after growth
- Scroll bars appear

**Prevention:**
- Plan maximum board size upfront
- Use relative positioning for pieces (grid coordinates, not pixels)
- Test growth from small → max size
- Consider scaling canvas down as board grows (keeps same viewport)

**Phase:** Board growth (Phase 4)

---

### 5. BroadcastChannel Message Order

**Problem:** Rapid config changes arrive out of order or overwhelm the game.

**Warning Signs:**
- Config appears to "jump around"
- Old values override new ones
- Performance drops during rapid changes

**Prevention:**
- Debounce admin inputs (50-100ms)
- Include timestamp in messages, ignore stale
- Apply config at defined sync points (not mid-frame)

**Phase:** Admin sync (Phase 3)

---

### 6. Freeze Cycle UX Confusion

**Problem:** Players don't understand why pieces stopped moving. Looks like a bug.

**Warning Signs:**
- Players refresh page during freeze
- Bug reports about "game stopped"

**Prevention:**
- Clear visual indicator during freeze (overlay, timer, color change)
- Countdown timer showing seconds until play resumes
- Distinct frozen state (pieces grayed out, "FROZEN" text)

**Phase:** Freeze mechanic (Phase 4)

---

### 7. Theme Hot-Swap Artifacts

**Problem:** Switching themes mid-game leaves visual artifacts or partial rendering.

**Warning Signs:**
- Old theme colors visible
- Pieces partially rendered in wrong colors
- Canvas not fully cleared

**Prevention:**
- Full canvas clear on theme change
- Redraw entire board state (not just current piece)
- Test theme switch at every game state

**Phase:** Theming (Phase 3)

---

### 8. Line Clear Animation Timing

**Problem:** Line clears without feedback, or animation blocks input.

**Warning Signs:**
- Players don't notice cleared lines
- Input feels laggy after clears
- Game stutters during clears

**Prevention:**
- Brief highlight before removing rows (100-200ms)
- Don't block game loop during animation
- Use flag to show "clearing" state vs actual board

**Phase:** Core mechanics (Phase 2)

---

## Medium-Risk Pitfalls

### 9. Keyboard Input Delay

**Problem:** Key presses feel unresponsive, especially for DAS (delayed auto-shift).

**Prevention:**
- Track keydown/keyup separately
- Implement repeat delay (150ms) then repeat rate (50ms)
- Process input every frame, not on events alone

---

### 10. Random Piece Generation Feels Unfair

**Problem:** True random can give 4+ of same piece in a row, frustrating players.

**Prevention:**
- Use "7-bag" system: shuffle all 7 pieces, deal, repeat
- Guarantees each piece appears exactly once per 7

---

### 11. Admin Tab Closed

**Problem:** Admin closes tab, game can't receive config updates.

**Prevention:**
- Game works standalone with defaults
- Config is stored locally as well as broadcast
- Reconnection not needed for same-browser case

---

## Low-Risk Pitfalls

### 12. Z-Fighting with Grid Lines

**Prevention:** Draw grid first, then pieces on top.

### 13. Memory Leak from Event Listeners

**Prevention:** Clean up listeners on game over/reset.

---

## v2.0 Feature-Specific Pitfalls

### 14. T-Spin Detection Edge Cases

**Problem:** T-spins not registering correctly, or false positives after wall kicks.

**Warning Signs:**
- T-spins counted after translation (not rotation)
- Mini vs full T-spin confusion
- Wrong point values

**Prevention:**
- Use 3-corner T algorithm (Tetris DS standard)
- Track last move type - only count T-spin if last move was rotation
- Full T-spin requires 2 front corners occupied
- Test near walls, after wall kicks

**Phase:** Scoring mechanics (Phase 5)

---

### 15. Combo Counter Reset Timing

**Problem:** Combos don't reset when expected, or reset too early.

**Warning Signs:**
- Infinite combos
- Combo resets mid-chain

**Prevention:**
- Reset combo counter when piece locks WITHOUT clearing lines
- Increment only on successful line clear
- Display combo for 2-3 seconds with fade

**Phase:** Scoring mechanics (Phase 5)

---

### 16. Audio Autoplay Blocked (CRITICAL)

**Problem:** Modern browsers block AudioContext creation without user gesture.

**Warning Signs:**
- No sound on first load
- Console errors about AudioContext
- Sound works after click but not before

**Prevention:**
- Create AudioContext on first user interaction (click/keypress)
- Add "Click to start" or detect first input
- Resume AudioContext on visibility change

**Phase:** Audio (Phase 6)

---

### 17. Sound Overlap/Distortion

**Problem:** Many sound events firing at once causes audio distortion.

**Warning Signs:**
- Distorted audio during combos/multi-clears
- Sound cuts out

**Prevention:**
- Use Web Audio API with gain nodes
- Limit concurrent sounds per category
- Priority system for overlapping sounds

**Phase:** Audio (Phase 6)

---

### 18. Key Rebinding Conflicts

**Problem:** Same key bound to multiple actions, or reserved keys causing issues.

**Warning Signs:**
- Game unplayable after rebind
- F5 refreshes page during rebind
- Actions fire twice

**Prevention:**
- Validate on rebind - warn if key in use
- Blacklist browser-reserved keys (F1-F12, Ctrl+combos)
- Use event.code (physical) not event.key (character)
- Save to localStorage immediately

**Phase:** Key remapping (Phase 7)

---

### 19. Integration with Freeze Cycle

**Problem:** New scoring mechanics don't respect freeze state.

**Warning Signs:**
- T-spins or combos process during freeze
- Game state corruption

**Prevention:**
- Check GameState enum before processing scoring
- Only score during PLAYING state
- Test all features during state transitions

**Phase:** All v2.0 phases

---

## v2.0 Prevention Summary

| Pitfall | Key Prevention | Phase |
|---------|----------------|-------|
| T-spin detection | 3-corner + last-move tracking | 5 |
| Combo reset | Reset on no-clear lock | 5 |
| Audio autoplay | User gesture initialization | 6 |
| Sound overlap | Web Audio + gain nodes | 6 |
| Key conflicts | Validation + blacklist | 7 |
| Freeze integration | Respect GameState enum | All |

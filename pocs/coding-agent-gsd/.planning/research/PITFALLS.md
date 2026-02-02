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

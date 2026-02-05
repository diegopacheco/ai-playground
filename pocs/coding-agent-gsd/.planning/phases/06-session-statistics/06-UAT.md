---
status: testing
phase: 06-session-statistics
source: [06-01-SUMMARY.md, 06-02-SUMMARY.md]
started: 2026-02-04T13:00:00Z
updated: 2026-02-04T13:00:00Z
---

## Current Test

number: 1
name: Stats Display in Sidebar
expected: |
  Sidebar shows LINES, TIME, PIECES, PPS, and APM stats.
  TIME displays in MM:SS format (e.g., "00:45").
  PPS shows decimal (e.g., "0.78"), APM shows whole number (e.g., "156").
awaiting: user response

## Tests

### 1. Stats Display in Sidebar
expected: Sidebar shows LINES, TIME, PIECES, PPS, and APM stats. TIME in MM:SS format, PPS as decimal, APM as whole number.
result: [pending]

### 2. Stats Update in Real-Time
expected: As you play, LINES increases when clearing rows, TIME counts up continuously, PIECES increases with each piece lock, PPS and APM update based on performance.
result: [pending]

### 3. Time Continues During Freeze
expected: During freeze cycle (10s frozen period), TIME continues counting while PIECES counter pauses. Time never stops.
result: [pending]

### 4. Stats Reset on Restart
expected: Press R to restart game. All stats reset to zero (LINES: 0, TIME: 00:00, PIECES: 0, PPS: 0.00, APM: 0).
result: [pending]

### 5. Session Summary on Game Over
expected: When game ends (pieces stack to top), dark overlay appears with "SESSION COMPLETE" title. Shows all 7 stats: Final Score, Lines Cleared, Level Reached, Session Time, Pieces Placed, PPS, APM. Footer says "Press R to play again".
result: [pending]

### 6. Session Summary Formatting
expected: Summary box is centered on screen with dark background. Stats are arranged in readable grid with labels and values. Box has cyan border.
result: [pending]

## Summary

total: 6
passed: 0
issues: 0
pending: 6
skipped: 0

## Gaps

[none yet]

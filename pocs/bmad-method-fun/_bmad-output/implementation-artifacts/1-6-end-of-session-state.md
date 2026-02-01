# Story 1.6: End-of-Session State

Status: ready-for-dev

## Story

As a player,
I want to see the end-of-session state when a game finishes,
so that I know the session has ended and can review the outcome.

## Acceptance Criteria

1. Given a session is running, when the game ends, then the session status changes to ended, and an end-of-session state is displayed to the player.

## Tasks / Subtasks

- [ ] Implement end-of-session condition (AC: 1)
  - [ ] End session after a configurable max placements
- [ ] Display end-of-session state (AC: 1)

## Dev Notes

- No Architecture or UX documents provided
- PRD constraints: SPA, target browsers Chrome/Firefox/Safari/Edge
- Performance targets: 60 FPS and input response under 50 ms

### Project Structure Notes

- No architecture guidance provided for file structure

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.6]
- [Source: _bmad-output/planning-artifacts/prd.md#Functional Requirements]
- [Source: _bmad-output/planning-artifacts/prd.md#Non-Functional Requirements]

## Dev Agent Record

### Agent Model Used

Codex CLI (GPT-5)

### Debug Log References

None

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created

### File List

- _bmad-output/implementation-artifacts/1-6-end-of-session-state.md

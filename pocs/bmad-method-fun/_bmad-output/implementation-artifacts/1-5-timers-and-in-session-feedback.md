# Story 1.5: Timers and In-Session Feedback

Status: review

## Story

As a player,
I want to see active timers and receive in-session feedback,
so that I can track time progression and gameplay changes.

## Acceptance Criteria

1. Given a session is running, when timers are active for board expansion and forced drops, then the player can see the active timers, and in-session feedback reflects score and level changes.

## Tasks / Subtasks

- [x] Implement timer display for board expansion and forced drop (AC: 1)
  - [x] Show active timer values in HUD
- [x] Show in-session feedback for score and level changes (AC: 1)

## Dev Notes

- No Architecture or UX documents provided
- PRD constraints: SPA, target browsers Chrome/Firefox/Safari/Edge
- Performance targets: 60 FPS and input response under 50 ms

### Project Structure Notes

- No architecture guidance provided for file structure

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 1.5]
- [Source: _bmad-output/planning-artifacts/prd.md#Functional Requirements]
- [Source: _bmad-output/planning-artifacts/prd.md#Non-Functional Requirements]

## Dev Agent Record

### Agent Model Used

Codex CLI (GPT-5)

### Debug Log References

None

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created
- Implemented HUD timers and feedback display
- Added timer visibility tests
- Tests: bun run test:run
- Lint: bun run lint

### File List

- frontend/src/App.tsx
- frontend/src/index.css
- frontend/src/App.test.tsx
- frontend/package.json
- frontend/bun.lock
- _bmad-output/implementation-artifacts/sprint-status.yaml
- _bmad-output/implementation-artifacts/1-5-timers-and-in-session-feedback.md

# Story 2.3: Update Timers

Status: review

## Story

As an admin,
I want to update the timers for board expansion and forced drops,
so that I can control gameplay pacing.

## Acceptance Criteria

1. Given I am viewing the admin configuration interface, when I change the board expansion timer or forced-drop timer, then the new timer values are saved, and the updated timing rules take effect in the game.

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

- [Source: _bmad-output/planning-artifacts/epics.md#Story 2.3]
- [Source: _bmad-output/planning-artifacts/prd.md#Functional Requirements]
- [Source: _bmad-output/planning-artifacts/prd.md#Non-Functional Requirements]

## Dev Agent Record

### Agent Model Used

Codex CLI (GPT-5)

### Debug Log References

None

### Completion Notes List

- Ultimate context engine analysis completed - comprehensive developer guide created
- Implemented admin timer inputs and live HUD updates
- Added timer settings test
- Tests: bun run test:run
- Lint: bun run lint

### File List

- frontend/src/App.tsx
- frontend/src/index.css
- frontend/src/App.test.tsx
- frontend/package.json
- frontend/bun.lock
- _bmad-output/implementation-artifacts/sprint-status.yaml
- _bmad-output/implementation-artifacts/2-3-update-timers.md

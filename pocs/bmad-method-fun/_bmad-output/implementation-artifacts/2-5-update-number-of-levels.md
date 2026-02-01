# Story 2.5: Update Number of Levels

Status: review

## Story

As an admin,
I want to change the number of levels,
so that I can control game length and progression.

## Acceptance Criteria

1. Given I am viewing the admin configuration interface, when I update the number of levels, then the new level count is saved, and the game uses the updated level count.

## Tasks / Subtasks

- [x] Implement level count input (AC: 1)
  - [x] Provide level count control
- [x] Apply updated level count to progression (AC: 1)

## Dev Notes

- No Architecture or UX documents provided
- PRD constraints: SPA, target browsers Chrome/Firefox/Safari/Edge
- Performance targets: 60 FPS and input response under 50 ms

### Project Structure Notes

- No architecture guidance provided for file structure

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 2.5]
- [Source: _bmad-output/planning-artifacts/prd.md#Functional Requirements]
- [Source: _bmad-output/planning-artifacts/prd.md#Non-Functional Requirements]

## Dev Agent Record

### Agent Model Used

Codex CLI (GPT-5)

### Debug Log References

None

### Completion Notes List

- Added admin control for number of levels with immediate impact on max placements
- Updated progression logic and tests to reflect configurable level count

### File List

- frontend/src/App.tsx
- frontend/src/App.test.tsx
- _bmad-output/implementation-artifacts/2-5-update-number-of-levels.md

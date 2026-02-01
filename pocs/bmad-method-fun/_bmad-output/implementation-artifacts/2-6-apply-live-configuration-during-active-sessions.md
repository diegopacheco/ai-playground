# Story 2.6: Apply Live Configuration During Active Sessions

Status: review

## Story

As an admin,
I want configuration changes to apply during active sessions,
so that I can tune gameplay without restarting the game.

## Acceptance Criteria

1. Given a session is running, when I apply configuration changes, then the changes are applied without restarting the session.
2. Gameplay reflects the updated settings.

## Tasks / Subtasks

- [x] Apply admin configuration changes during active sessions (AC: 1, 2)
  - [x] Ensure running sessions keep state while applying updates
- [x] Add tests covering live configuration updates (AC: 1, 2)

## Dev Notes

- SPA with React and Vite
- Admin configuration already exists in UI
- Performance targets: 60 FPS and input response under 50 ms

### Project Structure Notes

- Frontend logic in frontend/src/App.tsx
- Frontend tests in frontend/src/App.test.tsx

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 2.6]
- [Source: _bmad-output/planning-artifacts/prd.md#Functional Requirements]
- [Source: _bmad-output/planning-artifacts/prd.md#Non-Functional Requirements]

## Dev Agent Record

### Agent Model Used

Codex CLI (GPT-5)

### Debug Log References

None

### Completion Notes List

- Added test coverage for live configuration updates without restarting gameplay

### File List

- frontend/src/App.test.tsx
- _bmad-output/implementation-artifacts/2-6-apply-live-configuration-during-active-sessions.md

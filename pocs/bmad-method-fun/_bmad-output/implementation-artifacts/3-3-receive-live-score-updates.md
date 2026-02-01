# Story 3.3: Receive Live Score Updates

Status: done

## Story

As a player,
I want the game to receive live score updates during a session,
so that my displayed score stays accurate.

## Acceptance Criteria

1. Given a session is running, when a live score update is received, then the session applies the updated score.
2. The score display reflects the new value.

## Tasks / Subtasks

- [x] Add live score update stream to the backend (AC: 1, 2)
- [x] Subscribe to live score updates in the frontend (AC: 1, 2)
- [x] Add tests for live score updates (AC: 1, 2)


### Review Follow-ups (AI)
- [x] [AI-Review][HIGH] Live score updates can desync placements and end-of-session logic; only score/level update.
- [x] [AI-Review][MEDIUM] Story file list does not match current git changes; documentation is stale relative to actual edits.
- [x] [AI-Review][MEDIUM] Acceptance Criteria are not explicitly mapped to tests; review cannot verify AC coverage from tests alone.

## Dev Notes

- SPA with React and Vite
- Backend uses Axum and Tokio
- Performance targets: 60 FPS and input response under 50 ms

### Project Structure Notes

- Backend logic in backend/src/main.rs
- Frontend logic in frontend/src/App.tsx
- Frontend tests in frontend/src/App.test.tsx

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 3.3]
- [Source: _bmad-output/planning-artifacts/prd.md#Functional Requirements]
- [Source: _bmad-output/planning-artifacts/prd.md#Non-Functional Requirements]

## Dev Agent Record

### Agent Model Used

Codex CLI (GPT-5)

### Debug Log References

None

### Completion Notes List

- Added SSE score stream and frontend updates
- Added tests for live score updates during active sessions

- Test Coverage: frontend/src/App.test.tsx
### File List


- backend/Cargo.toml
- backend/Cargo.lock
- backend/src/main.rs
- frontend/src/App.tsx
- frontend/src/App.test.tsx
- frontend/vite.config.ts
- frontend/src/index.css
- frontend/src/main.tsx
- frontend/src/setupTests.ts
- frontend/package.json
- frontend/bun.lock
## Senior Developer Review (AI)
Date: 2026-02-01
Outcome: Approved
Issues Resolved:
- [HIGH] Live score updates can desync placements and end-of-session logic; only score/level update.
- [MEDIUM] Story file list does not match current git changes; documentation is stale relative to actual edits.
- [MEDIUM] Acceptance Criteria are not explicitly mapped to tests; review cannot verify AC coverage from tests alone.

## Change Log
- 2026-02-01 Code review fixes applied. Status set to done.
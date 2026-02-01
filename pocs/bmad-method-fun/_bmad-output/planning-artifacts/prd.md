---
stepsCompleted:
- step-01-init
- step-02-discovery
- step-03-success
- step-04-journeys
- step-05-domain
- step-06-innovation
- step-07-project-type
- step-08-scoping
- step-09-functional
- step-10-nonfunctional
- step-11-polish
inputDocuments: []
documentCounts:
  productBriefs: 0
  research: 0
  brainstorming: 0
  projectDocs: 0
workflowType: prd
classification:
  projectType: game (web)
  domain: gaming
  complexity: low
  projectContext: greenfield
---
# Product Requirements Document - bmad-method-fun

**Author:** Diegopacheco
**Date:** 2026-02-01T08:15:21Z

## Success Criteria

### User Success

Players reach 100 points. Each good move earns 10 points.

### Business Success

Release now as a public web game. Success is measured by release completion and availability.

### Technical Success

Performance and reliability targets are defined in Non-Functional Requirements.

### Measurable Outcomes

- Players can reach 100 points with the scoring rule: 10 points per good move.

## Product Scope

### MVP - Minimum Viable Product

All desired features included in the first release.

### Growth Features (Post-MVP)

None specified.

### Vision (Future)

None specified.

## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**MVP Approach:** Experience MVP
**Resource Requirements:** Single developer with web game + UI + basic backend skills

### MVP Feature Set (Phase 1)

**Core User Journeys Supported:**
- Player happy path and edge case
- Admin live configuration

**Must-Have Capabilities:**
- Core gameplay loop (timed falling blocks, scoring, levels)
- Admin runtime controls (backgrounds, timers, difficulty, number of levels)
- UI for player and admin
- SSE for live updates (admin config, timers, score events)

### Post-MVP Features

**Phase 2 (Post-MVP):**
- None

**Phase 3 (Expansion):**
- Sound effects during gameplay

### Risk Mitigation Strategy

**Technical Risks:** Timing accuracy and performance under SPA + SSE
**Market Risks:** Competition from established casual games like Bejeweled
**Resource Risks:** Solo implementation bandwidth

## User Journeys

### Player Journey (Happy Path)
A 20 to 40 year old player opens the game on a plane, looking for a short burst of fun. They land on a simple start screen, begin a session immediately, and settle into a smooth flow. The board expands every 30 seconds, and every 40 seconds a piece falls automatically without control, adding tension. They adapt to rising difficulty, level up, and chase points (10 per good move). Their session ends when they choose to stop or lose, but they feel satisfied because the experience is smooth and responsive.

### Player Journey (Edge Case)
The same player tries to play but the game feels slow and buggy. Inputs lag, drops stutter, and the pacing breaks the rhythm. They quit early, frustrated, and are unlikely to return until performance improves.

### Admin Journey
You open the admin controls to tune the game live. You adjust level backgrounds, timers, difficulty, and number of levels without restarting the game. The changes apply immediately to active sessions, and you can verify the effect in real time. The admin flow feels reliable and quick, giving you confidence to adjust gameplay balance on the fly.

### Journey Requirements Summary
- Fast start into gameplay with minimal friction.
- Smooth, responsive play session with consistent timing.
- Visible, predictable level progression tied to time and scoring.
- Robust performance to avoid lag or stutter.
- Admin runtime controls for level backgrounds, timers, difficulty, and level count.
- Live configuration changes applied without restarting sessions.

## Web App Specific Requirements

### Project-Type Overview
Single-page web game with timed mechanics and live runtime configuration. No SEO requirements.

### Technical Architecture Considerations
- SPA architecture with client-rendered UI.
- SSE channel for live admin config updates, timers, and score events.
- Target browsers: Chrome, Firefox, Safari, Edge.

### Dynamic Requirements
- Responsive layout to handle expanding board area.
- Accessibility not required beyond basic usability.

### Implementation Considerations
- Keep latency low for timed events and automatic drops.
- Ensure deterministic timing for board expansion (30s) and forced piece drop (40s).

## Functional Requirements

### Gameplay Session
- FR1: Player can start a new game session.
- FR2: Player can end a game session.
- FR3: Player can view current session status (running, paused, ended).

### Board & Pieces
- FR4: Game can render a playable board area for the session.
- FR5: Game can expand the board at a fixed interval during a session.
- FR6: Game can introduce a falling piece into the board.
- FR7: Game can apply a forced piece drop at a fixed interval without player control.
- FR8: Game can detect and resolve piece placement outcomes.

### Scoring & Levels
- FR9: Game can award points for a good move.
- FR10: Game can calculate total score for the session.
- FR11: Game can increment level based on defined progression.
- FR12: Player can view current score and level during a session.

### Timers & Progression
- FR13: Game can track elapsed session time.
- FR14: Game can apply timed board expansion logic.
- FR15: Game can apply timed forced-drop logic.
- FR16: Player can view active timers during a session.

### UI & Feedback
- FR17: Player can access a start screen before gameplay.
- FR18: Player can see in-session feedback for score and level changes.
- FR19: Player can see end-of-session state.

### Admin Configuration
- FR20: Admin can access a configuration interface.
- FR21: Admin can change level backgrounds.
- FR22: Admin can change timers for board expansion and forced drops.
- FR23: Admin can change difficulty settings.
- FR24: Admin can change the number of levels.
- FR25: Admin can apply configuration changes during active sessions.
- FR26: Game can apply admin configuration changes at runtime.

### Live Updates
- FR27: Game can receive live configuration updates during a session.
- FR28: Game can receive live timer updates during a session.
- FR29: Game can receive live score updates during a session.

### Reliability
- FR30: Game can recover from a temporary update interruption without losing the session.

## Non-Functional Requirements

### Performance
- Maintain 60 FPS during gameplay on target browsers.
- Player input response time under 50 ms.

### Reliability
- No more than 1 dropped live update per session.

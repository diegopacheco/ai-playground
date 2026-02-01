---
stepsCompleted:
- step-01-validate-prerequisites
- step-02-design-epics
- step-03-create-stories
inputDocuments:
- /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/bmad-method-fun/_bmad-output/planning-artifacts/prd.md
---

# bmad-method-fun - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for bmad-method-fun, decomposing the requirements from the PRD, UX Design if it exists, and Architecture requirements into implementable stories.

## Requirements Inventory

### Functional Requirements

1. FR1: Player can start a new game session.
2. FR2: Player can end a game session.
3. FR3: Player can view current session status (running, paused, ended).
4. FR4: Game can render a playable board area for the session.
5. FR5: Game can expand the board at a fixed interval during a session.
6. FR6: Game can introduce a falling piece into the board.
7. FR7: Game can apply a forced piece drop at a fixed interval without player control.
8. FR8: Game can detect and resolve piece placement outcomes.
9. FR9: Game can award points for a good move.
10. FR10: Game can calculate total score for the session.
11. FR11: Game can increment level based on defined progression.
12. FR12: Player can view current score and level during a session.
13. FR13: Game can track elapsed session time.
14. FR14: Game can apply timed board expansion logic.
15. FR15: Game can apply timed forced-drop logic.
16. FR16: Player can view active timers during a session.
17. FR17: Player can access a start screen before gameplay.
18. FR18: Player can see in-session feedback for score and level changes.
19. FR19: Player can see end-of-session state.
20. FR20: Admin can access a configuration interface.
21. FR21: Admin can change level backgrounds.
22. FR22: Admin can change timers for board expansion and forced drops.
23. FR23: Admin can change difficulty settings.
24. FR24: Admin can change the number of levels.
25. FR25: Admin can apply configuration changes during active sessions.
26. FR26: Game can apply admin configuration changes at runtime.
27. FR27: Game can receive live configuration updates during a session.
28. FR28: Game can receive live timer updates during a session.
29. FR29: Game can receive live score updates during a session.
30. FR30: Game can recover from a temporary update interruption without losing the session.

### NonFunctional Requirements

1. Maintain 60 FPS during gameplay on target browsers.
2. Player input response time under 50 ms.
3. No more than 1 dropped live update per session.

### Additional Requirements

- None (no Architecture or UX documents provided)

### FR Coverage Map



## Epic List









**Acceptance Criteria:**




## Epic 1: Core Gameplay Loop

Players can start and complete a full game session with board expansion, forced drops, and scoring.

### Story 1.1: Start Game Session and Status

As a player,
I want to start a new game session and see its current status,
So that I know when gameplay begins and whether it is running, paused, or ended.

**Acceptance Criteria:**

**Given** the game is at the start screen
**When** the player starts a new game
**Then** a new session begins and status is set to running
**And** the current session status is visible to the player

### Story 1.2: Render Board and Introduce Falling Piece

As a player,
I want the game to render a playable board and introduce a falling piece,
So that I can begin interacting with the gameplay.

**Acceptance Criteria:**

**Given** a session is running
**When** gameplay begins
**Then** a playable board is displayed
**And** an initial falling piece appears on the board

### Story 1.3: Forced Drop and Placement Resolution

As a player,
I want the game to force a piece drop at a fixed interval and resolve the placement,
So that the game progresses even without my control.

**Acceptance Criteria:**

**Given** a session is running and a piece is active
**When** the forced-drop interval occurs
**Then** the piece drops without player control
**And** the game resolves the placement outcome on the board

### Story 1.4: Scoring and Level Progression

As a player,
I want the game to award points for good moves and advance my level over time,
So that I can track progress and feel increasing challenge.

**Acceptance Criteria:**

**Given** a session is running
**When** I make a good move
**Then** 10 points are added to my score
**And** the current score is updated
**When** the level progression condition is met
**Then** the level increases and is visible to the player

### Story 1.5: Timers and In-Session Feedback

As a player,
I want to see active timers and receive in-session feedback,
So that I can track time progression and gameplay changes.

**Acceptance Criteria:**

**Given** a session is running
**When** timers are active for board expansion and forced drops
**Then** the player can see the active timers
**And** in-session feedback reflects score and level changes

### Story 1.6: End-of-Session State

As a player,
I want to see the end-of-session state when a game finishes,
So that I know the session has ended and can review the outcome.

**Acceptance Criteria:**

**Given** a session is running
**When** the game ends
**Then** the session status changes to ended
**And** an end-of-session state is displayed to the player

## Epic 2: Admin Live Configuration

Admin can configure gameplay settings live without restarting sessions.

### Story 2.1: Access Admin Configuration Interface

As an admin,
I want to access a configuration interface,
So that I can manage gameplay settings.

**Acceptance Criteria:**

**Given** I am an admin
**When** I open the admin configuration interface
**Then** I can view the available gameplay settings
**And** the interface is ready to accept changes

### Story 2.2: Update Level Backgrounds

As an admin,
I want to change level backgrounds,
So that I can adjust the visual experience of the game.

**Acceptance Criteria:**

**Given** I am viewing the admin configuration interface
**When** I update the level background setting
**Then** the selected background is saved
**And** the game reflects the updated background setting

### Story 2.3: Update Timers

As an admin,
I want to update the timers for board expansion and forced drops,
So that I can control gameplay pacing.

**Acceptance Criteria:**

**Given** I am viewing the admin configuration interface
**When** I change the board expansion timer or forced-drop timer
**Then** the new timer values are saved
**And** the updated timing rules take effect in the game

### Story 2.4: Update Difficulty Settings

As an admin,
I want to update difficulty settings,
So that I can tune the challenge of the game.

**Acceptance Criteria:**

**Given** I am viewing the admin configuration interface
**When** I change the difficulty settings
**Then** the new difficulty values are saved
**And** the updated difficulty takes effect in the game

### Story 2.5: Update Number of Levels

As an admin,
I want to change the number of levels,
So that I can control game length and progression.

**Acceptance Criteria:**

**Given** I am viewing the admin configuration interface
**When** I update the number of levels
**Then** the new level count is saved
**And** the game uses the updated level count

### Story 2.6: Apply Live Configuration During Active Sessions

As an admin,
I want configuration changes to apply during active sessions,
So that I can tune gameplay without restarting the game.

**Acceptance Criteria:**

**Given** a session is running
**When** I apply configuration changes
**Then** the changes are applied without restarting the session
**And** gameplay reflects the updated settings

## Epic 3: Live Updates & Resilience

Game can receive live updates reliably during sessions and recover from interruptions.

### Story 3.1: Receive Live Configuration Updates

As a player,
I want the game to receive live configuration updates during a session,
So that gameplay reflects the latest settings.

**Acceptance Criteria:**

**Given** a session is running
**When** a live configuration update is received
**Then** the session applies the updated configuration
**And** gameplay reflects the new settings

### Story 3.2: Receive Live Timer Updates

As a player,
I want the game to receive live timer updates during a session,
So that timing rules reflect the latest settings.

**Acceptance Criteria:**

**Given** a session is running
**When** a live timer update is received
**Then** the session applies the updated timing rules
**And** active timers reflect the new values

### Story 3.3: Receive Live Score Updates

As a player,
I want the game to receive live score updates during a session,
So that my displayed score stays accurate.

**Acceptance Criteria:**

**Given** a session is running
**When** a live score update is received
**Then** the session updates the displayed score
**And** the score reflects the latest value

### Story 3.4: Recover from Update Interruptions

As a player,
I want the game to recover from a temporary update interruption,
So that my session continues without being lost.

**Acceptance Criteria:**

**Given** a session is running
**When** a live update interruption occurs
**Then** the game continues the session without losing progress
**And** updates resume when connectivity is restored

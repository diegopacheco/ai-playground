# terminal-rpg-agent Design Doc

## Goal

`terminal-rpg-agent` turns learning tasks into a visual RPG. The user chooses a theme, answers questions, gains score and XP, collects inventory, survives failures, and fights bosses. Every run starts a new game. The selected theme fully scopes the quests. The browser game is the primary experience. The terminal runner is the fallback for real shell command quests.

## Non Goals

This first version will not create a remote service, account system, multiplayer mode, or external database. It runs locally as static browser files plus a shell fallback.

## User Command

Primary command:

```text
/rpg-learn $THEME
```

If `$THEME` is missing, the agent asks the user to choose one theme using numeric options:

```text
1. Algorithms
2. Data Structures
3. Generative AI
4. Machine Learning
5. SRE/DevOps
6. Management
```

Accepted theme aliases:

```text
algorithms, algorithm, algo
datastructures, data-structures, data structures, ds
generative-ai, generative ai, genai, ai
machine-learning, machine learning, ml
sre-devops, sre, devops
management, manager, leadership
```

## Experience

The skill generates a short RPG session based on the selected theme. A session never mixes themes. If the user picks Algorithms, every question, reward, command quest, and boss fight is about Algorithms.

Each session contains:

1. Character intro
2. Theme-specific map
3. Quest cards
4. Multiple-choice questions
5. Occasional command-based quests
6. Inventory rewards
7. Failure effects
8. XP and level progression
9. Boss fight
10. Final score screen

The game must be visual in the browser. The visual style uses cards, progress bars, compact stat panels, animated feedback, inventory, boss health, and per-theme history. The terminal fallback must still be readable in plain text.

## Themes

### Algorithms

Focus areas:

1. Big O
2. Sorting
3. Searching
4. Recursion
5. Dynamic programming
6. Graph traversal
7. Greedy choices

Boss fight ideas:

1. Complexity Hydra
2. Recursion Labyrinth
3. Graph Warden

### Data Structures

Focus areas:

1. Arrays
2. Linked lists
3. Stacks
4. Queues
5. Hash maps
6. Trees
7. Heaps
8. Graphs

Boss fight ideas:

1. Hash Collision Beast
2. Heap Guardian
3. Tree Rotator

### Generative AI

Focus areas:

1. Prompt design
2. Context windows
3. Retrieval
4. Tool use
5. Agents
6. Evaluation
7. Safety

Boss fight ideas:

1. Hallucination Phantom
2. Context Kraken
3. Eval Sentinel

### Machine Learning

Focus areas:

1. Supervised learning
2. Unsupervised learning
3. Features
4. Training and validation
5. Overfitting
6. Metrics
7. Model selection

Boss fight ideas:

1. Overfit Dragon
2. Gradient Golem
3. Metric Mimic

### SRE/DevOps

Focus areas:

1. Linux basics
2. Observability
3. CI
4. Containers
5. Kubernetes
6. Incident response
7. Reliability

Boss fight ideas:

1. Pager Storm
2. CrashLoop Overlord
3. Latency Lich

### Management

Focus areas:

1. Prioritization
2. Feedback
3. Delegation
4. Stakeholder alignment
5. Planning
6. Conflict resolution
7. Hiring and team health

Boss fight ideas:

1. Scope Creep Titan
2. Meeting Maze
3. Priority Chimera

## Game Loop

Each run follows this loop:

1. Resolve theme
2. Create or load player state
3. Generate a session plan
4. Render intro card
5. Ask quest question
6. Evaluate answer
7. Update score, XP, health, inventory, and quest state
8. Render result card
9. Continue until the boss unlocks
10. Run boss fight
11. Persist final state
12. Render final score

## Quest Types

### Knowledge Quest

The user answers a multiple-choice question. The agent validates the answer and explains the correct reasoning briefly.

### Scenario Quest

The user chooses the best action in a realistic theme-specific scenario. These should test judgment rather than memorization.

### Command Quest

The agent asks the user for permission to run or inspect a real command when useful. The command must be directly connected to the theme and must be safe by default.

SRE/DevOps command quests can inspect files, run local status commands, parse logs, or validate shell output. Algorithms and data structures command quests can run local test scripts later if the implementation includes generated exercises.

### Boss Quest

The boss fight is a sequence of harder questions. Boss health decreases on correct answers. Player health decreases on wrong answers. Inventory can provide one-time benefits.

## State Model

Each run creates a fresh game state. The implementation can store completed run summaries by theme, but it must not load previous XP, health, inventory, or score into a new run.

Browser run history should be stored in local storage by theme. Terminal fallback run history should be stored locally in a small per-theme file under a user state directory. The implementation should avoid external dependencies.

Recommended in-memory state fields:

```text
player_name
theme
level
xp
score
health
max_health
inventory
completed_quests
failed_quests
bosses_defeated
last_played_at
streak
```

Recommended history fields:

```text
run_id
theme
level
xp
score
health
boss_defeated
completed_at
```

History should be recoverable if the file is missing or invalid. Invalid history should not block a new game.

## Scoring

Base scoring:

```text
correct answer: +100 score, +25 XP
wrong answer: -10 health
boss correct answer: +150 score, +40 XP, boss health -25
boss wrong answer: -20 health
quest completion streak: +25 score per streak level
level up: every 100 XP
```

XP, health, score, and inventory reset at the start of each run.

## Inventory

Inventory items should be theme-flavored and mechanically simple.

Item effects:

```text
Hint Token: reveal a hint before answering
Shield: prevent one health loss
Double XP: double XP for one quest
Retry Charm: retry one failed question
Boss Key: unlock boss fight early
```

The first version should keep inventory automatic. Items are awarded and consumed by clear prompts without complex menus.

## Visual Design

The browser game should use a card layout with:

1. Title bar
2. Theme badge
3. Player stats
4. Quest prompt
5. Answer options
6. Progress bar
7. Inventory row
8. Result card after each answer
9. Boss health bar

Animation strategy:

1. Animate card hits and successful attacks.
2. Use progress bars for HP, XP, boss health, and map progress.
3. Keep the browser game dependency-free.
4. Keep the plain text fallback polished.

Card style should be cool but compact. The game should look richer than plain Q&A, while staying easy to read in an agent conversation.

## Skill Structure

Implemented files:

```text
SKILL.md
scripts/rpg_learn.sh
web/index.html
web/styles.css
web/game.js
agents/openai.yaml
install.sh
uninstall.sh
README.md
design-doc.md
```

The browser game owns the visual RPG experience. The shell runner keeps a terminal fallback for users who need real shell execution.

## Skill Behavior

The skill should instruct the agent to:

1. Detect `/rpg-learn`.
2. Parse the optional theme.
3. Ask for a numeric theme choice when missing or invalid.
4. Point to the browser game as the primary RPG surface.
5. Generate a fresh session for the selected theme.
6. Persist run history locally by theme.
7. Use command execution only through the terminal fallback.
8. Keep score, XP, HP, inventory, and boss health visible.

## Install Flow

`install.sh` should support three choices:

```text
1. Codex
2. Claude
3. Both
```

The script should copy the skill into the correct user-level skill directory for the selected agent.

Codex target:

```text
~/.codex/skills/terminal-rpg-agent
```

Claude target:

```text
~/.claude/skills/terminal-rpg-agent
```

The installer should:

1. Ask which agent to install for.
2. Create the target directories.
3. Copy the skill files.
4. Avoid overwriting without confirming.
5. Print the installed paths.
6. Exit nonzero on failure.

The uninstaller should:

1. Ask which agent to remove from.
2. Confirm removal.
3. Remove only `terminal-rpg-agent`.
4. Leave user state unless the user confirms state deletion.
5. Print what was removed.

## Safety

Command quests must follow these rules:

1. Prefer read-only commands.
2. Explain the command before execution.
3. Do not run destructive commands.
4. Do not install dependencies automatically.
5. Do not send local files or secrets to remote services.
6. Do not inspect unrelated user directories.
7. Keep generated shell commands simple.

## Implementation Plan

### Phase 1

Create the skill shell:

1. `SKILL.md`
2. Theme resolution
3. Basic game loop
4. Question generation rules
5. Card templates

### Phase 2

Add per-theme run history:

1. JSON state file
2. XP and level calculation
3. Inventory
4. Streaks
5. Boss tracking

### Phase 3

Add installer scripts:

1. `install.sh`
2. `uninstall.sh`
3. Codex install path
4. Claude install path
5. Both-agent flow

### Phase 4

Add polish:

1. Better visual cards
2. Theme-specific boss fights
3. Command quests
4. Failure states
5. Final score screen

### Phase 5

Add tests:

1. Install path validation
2. Uninstall path validation
3. Theme parsing
4. State recovery
5. Basic shell script checks

## Validation

Validation should include:

```text
./install.sh
./uninstall.sh
shellcheck scripts/*.sh when available
/rpg-learn algorithms
/rpg-learn
```

If `shellcheck` is not installed, the test output should state that clearly.

## Decisions

1. Every run starts a new game.
2. Run history is separated by theme.
3. Quests, command quests, and boss fights are scoped to the selected theme.
4. The first implementation includes skill instructions, the RPG runner, install flow, uninstall flow, and validation.

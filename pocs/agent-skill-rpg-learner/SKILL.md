---
name: terminal-rpg-agent
description: Run themed RPG learning sessions from `/rpg-learn` requests. Use when the user wants a visual browser game or terminal fallback with quests, cards, XP, score, inventory, failures, boss fights, and safe shell command quests for Algorithms, Data Structures, Generative AI, Machine Learning, SRE/DevOps, or Management.
---

# terminal-rpg-agent

Use the browser game as the primary experience for `/rpg-learn` requests. Use the bundled terminal runner only when the user asks to play fully inside the terminal.

## Workflow

1. If the user provides a theme, give them the local `web/index.html?theme=$THEME` path.
2. If the user does not provide a theme, give them the local `web/index.html` path and list the six numeric theme choices.
3. Keep the selected theme scoped for the whole session.
4. Do not mix questions from other themes.
5. Treat every run as a new game.
6. Let the browser game persist only per-theme run history in local storage.
7. For real shell command quests, use `scripts/rpg_learn.sh` as the terminal fallback.
8. Do not reduce the game to plain chat questions unless the user asks for terminal mode.

## Browser Game

Open:

```text
web/index.html
```

Open a selected theme:

```text
web/index.html?theme=algorithms
```

## Theme Command

```bash
scripts/rpg_learn.sh algorithms
```

Valid themes are:

```text
algorithms
datastructures
generative-ai
machine-learning
sre-devops
management
```

## Missing Theme

Run:

```bash
scripts/rpg_learn.sh
```

The runner asks the user to choose:

```text
1. Algorithms
2. Data Structures
3. Generative AI
4. Machine Learning
5. SRE/DevOps
6. Management
```

## Safety

Only run the fixed commands embedded in `scripts/rpg_learn.sh`. Do not invent destructive command quests. Do not install dependencies. Do not inspect unrelated user directories. Keep the session visual, concise, and theme-specific.

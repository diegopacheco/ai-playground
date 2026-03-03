# Ghost

https://github.com/adamveld12/ghost

Commit intentions, not code.

Ghost is a CLI that flips the git workflow: instead of committing code, you commit prompts. An AI coding agent generates the artifacts; the commit captures both the intent and the output. Your git history becomes a chain of prompts + their results.

Supports claude, gemini, codex, and opencode — swap agents per-commit or set a default.

## Experience Notes

* Installion was easy
* Did not like Opus was not the default model for claude
* 

## Result

```
ghost commit -m "create a memory game with frontend written in React, Tanstack, vite, bun and the backend in rust with sqllite. make sure there is a leaderboard ui, there is a timmer for 120s to finish the game and the game tracks how many moves."
```

```
  ▸ ghost
  agent     claude
  model     claude-sonnet-4-6
  intent    create a memory game with frontend written in React, Tanstack, vite, bun and the backend in rust with sqllite. make sure there is a leaderboard ui, there is a timmer for 120s to finish the game and the game tracks how many moves.

⠙ running claude…
```
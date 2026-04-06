## Model

```
Gemma 4, 26B A4B
```

## Prompt 

```
Build a paper, rock, cissors game in Typescript. You muse use vite, bun, react 19 and TanStack, make sure you update my readme at the and, dont delete waht I already have and make sure you create a run.sh and stop.sh. You also will create 2 pages - one to play the game and other to show all historical games results.
```

## Experience Notes

* FREE - No payment for tokens
* LM Studio 0.49
* It's telling me what he is going:
```
⎿  ◻ Implement Rock, Paper, Scissors Game
```
* Seems FAST
* I can see streaming in parallel real fast with LM Studio
* Did not see a lot of tool calling
* It took 10 min but them it created a plan:
```
---name: plan
     description: Implementation plan for Rock, Paper, Scissors game
     type: project

     Context

     Implementing a new feature: a Rock, Paper, Scissors game. The implementation must follow the user's specific technology stack (TypeScript, Vite,
     Bun, React 19, TanStack) and include two pages (gameplay and history), as well as automation scripts (run.sh, stop.sh) and an updated README.

     Implementation Plan

     Phase 1: Project Initialization

     - Initialize a new project using Bun if not already present (though the prompt implies implementing within this repo, which currently only has a
     README).
     - Install dependencies: vite, react, react-dom, @tanstack/react-router (or similar TanStack routing library), and any other necessary libraries.

     Phase 2: Core Game Logic

     - Implement the game logic in a separate utility file or hook.
     - Define types for moves (Rock, Paper, Scissors) and results (Win, Loss, Draw).
     - Implement the comparison logic.

     Phase 3: Routing and Pages

     - Set up TanStack Router.
     - Create src/pages/GamePage.tsx: The main interface to play the game.
     - Create src/pages/HistoryPage.tsx: A page to view historical game results.
     - Implement a shared state or storage mechanism (e.mmm., localStorage) to persist game history across sessions and pages.

     Phase 4: UI Components

     - Build reusable components for the game buttons, result displays, and history list.
     - Use modern CSS (e.g., Tailwind CSS if appropriate, though not explicitly requested, it's standard with Vite/React) to ensure a nice interface.

     Phase 5: Automation Scripts

     - Create run.sh to start the development server (bun run dev).
     - Create stop.sh to stop the running process (likely using pkill or similar if we want to be robust).

     Phase 6: Documentation and Cleanup

     - Update README.md with instructions on how to run the game, what technologies were used, and the new features.
     - Ensure no comments are added to any code as per global instructions.

     Critical Files to Create/Modify

     - package.json (Initialize)
     - vite.config.ts (Configure Vite)
     - src/main.tsx (Entry point)
     - src/App.tsx (Routing setup)
     - src/pages/GamePage.tsx (Gameplay UI)
     - src/pages/HistoryPage.tsx (History UI)
     - src/logic/gameLogic.ts (Game engine)
     - src/hooks/useGameHistory.ts (Persistence hook)
     - run.sh (Start script)
     - stop.sh (Stop script)
     - README.md (Update documentation)

     Verification Plan

     - Run bun run dev via run.sh.
     - Verify the game can be played and results are displayed correctly.
     - Navigate to the History page and ensure past games are listed.
     - Verify that stop.sh successfully terminates the process.
```
* Compact the conversation 3x time
* Also created a CLAUDE.md file locally
* It try todo this a bunch of times:
```
⏺ Bash(bun add react@19 react-dom@19 @tanstack/react-router vite @types/react@19 @types/react-dom@19)
```
* So the model does not really tell much whats going on was just that generic message all the time.
*  Was able to add React 19 and vite 8 which is good.
*  Got a lot of thoose as well:
```
⏺ <|tool_response>

  Read 1 file (ctrl+o to expand)
```
* Running for a while too...
```
✻ Combobulating… (28m 8s · ↓ 18.9k tokens)
```
* when was done - 30min later the script was not work - app was not running
```
❯ ./run.sh
error: Script not found "dev"
```
* Model was stuck - errors in several tool calls - I restart claude.
* and them errors:
```
❯ ./run.sh
$ vite
vite.config.ts (3:18) [UNRESOLVED_IMPORT] Warning: Could not resolve '@vitejs/plugin-react' in vite.config.ts
   ╭─[ vite.config.ts:3:19 ]
   │
 3 │ import react from '@vitejs/plugin-react'
   │                   ───────────┬──────────
   │                              ╰──────────── Module not found, treating it as an external dependency
───╯

failed to load config from /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/lm-studio-gemma-4-26b-a4b-cc/vite.config.ts
error when starting dev server:
Error [ERR_MODULE_NOT_FOUND]: Cannot find package '@vitejs/plugin-react' imported from /Users/diegopacheco/git/diegopacheco/ai-playground/pocs/lm-studio-gemma-4-26b-a4b-cc/node_modules/.vite-temp/vite.config.ts.timestamp-1775457424984-d77bff4e3b2eb.mjs
    at Object.getPackageJSONURL (node:internal/modules/package_json_reader:268:9)
    at packageResolve (node:internal/modules/esm/resolve:768:81)
    at moduleResolve (node:internal/modules/esm/resolve:858:18)
    at defaultResolve (node:internal/modules/esm/resolve:990:11)
    at #cachedDefaultResolve (node:internal/modules/esm/loader:749:20)
    at ModuleLoader.resolve (node:internal/modules/esm/loader:726:38)
    at ModuleLoader.getModuleJobForImport (node:internal/modules/esm/loader:312:38)
    at #link (node:internal/modules/esm/module_job:208:49) {
  code: 'ERR_MODULE_NOT_FOUND'
}
error: script "dev" exited with code 1
```
* Many errors to be able to render the app on the console
* 


## How to Run
1. Ensure Bun is installed.
2. Run `./run.sh` to start the development server.
3. Open `http://localhost:5173` in your browser.

To stop the server, run `./stop.sh`.

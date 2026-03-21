# Result

Original source code: https://github.com/diegopacheco/ai-playground/tree/main/pocs/agent-werewolf

claude
```
 /bruno-generator
```

```
Bruno Collection Generated
===========================
Project:     werewolf-server
Framework:   Rust / Actix-web
Base URL:    http://localhost:3000
Output:      bruno-collection/

Endpoints Found:
GET:    4
POST:   1
PUT:    0
DELETE: 0
PATCH:  0
Total:  5

Folders:
games/  (4 endpoints)
agents/ (1 endpoint)

Files Generated: 9

To use: Open bruno-collection/ folder in Bruno ap
```

## Bruno Collection

The generated Bruno collection is available at [bruno-collection/](bruno-collection/). It contains:

* `bruno.json` - Bruno project configuration
* `agents/`
  * `get-all-agents.bru` - GET all available AI agents
* `games/`
  * `get-all-games.bru` - GET all werewolf games
  * `get-game-by-id.bru` - GET a single game by UUID
  * `create-game.bru` - POST create a new game
  * `get-game-stream.bru` - GET stream game events

To use it, open the `bruno-collection/` folder in the Bruno app.

## Screenshots

### List Available Agents
![List Available Agents](Screenshot%202026-03-20%20at%2014.40.07.png)

Bruno executing a GET request to `http://localhost:3000/api/agents`. The response returns a JSON array of available AI agents (Claude, Gemini, Copilot, Codex) with their supported models and default model configuration. Status 200 OK.

### List All Games
![List All Games](Screenshot%202026-03-20%20at%2014.40.23.png)

Bruno executing a GET request to `http://localhost:3000/api/games`. The response returns a JSON array of all werewolf games with their full state including game id, status, players, agents, roles, votes, and timestamps. The left sidebar shows the full collection structure with agents and games folders.

### Get Game By Id
![Get Game By Id](Screenshot%202026-03-20%20at%2014.40.58.png)

Bruno executing a GET request to `http://localhost:3000/api/games/{id}` to fetch a single game by its UUID. The response returns the complete game object with all agent details including agent names, models, roles (villager), alive status, and vote counts.

### Create Game
![Create Game](Screenshot%202026-03-20%20at%2014.41.29.png)

Bruno executing a POST request to `http://localhost:3000/api/games` with a JSON body containing an array of agents (Claude, Gemini, Copilot, Codex) each with their model configuration. The response returns the newly created game with its generated id and status set to "running". Status 200 OK.

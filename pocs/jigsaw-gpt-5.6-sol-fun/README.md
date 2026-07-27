# Big Cat Jigsaw

A light-themed, real-time collaborative jigsaw with nine recursive levels of 315 pieces each. Every browser receives a stable player identity, live cursors show where everyone is working, and piece movement is synchronized through server-sent events.

## Requirements

- Node.js 20 or newer
- curl

## Start

```bash
./start.sh
```

Open `http://localhost:4177` in two different browsers to join as two players.

## Stop

```bash
./stop.sh
```

## Test

```bash
./test.sh
```

The test starts the server, checks its health response, loads the page, joins a browser player, and verifies the nine-level 315-piece room.

## How it works

- The Node.js server uses only built-in modules.
- A browser identity is stored in local storage.
- Server-sent events deliver players, cursors, progress, and piece updates.
- A piece snaps into place only when dropped close to its correct position.
- The board uses a 21 by 15 grid matching the artwork's 7:5 ratio.
- Completing a board opens the next nested world for every connected player.
- The ninth board leads back to the first board and begins a deeper loop.

## The nine worlds

1. Cats around a cozy bed
2. An endless library
3. An ocean above and below the waves
4. A sandcastle kingdom
5. A daylight carnival
6. A rainy city
7. A secret garden after rain
8. A floating observatory
9. Cats assembling the puzzle within the puzzle

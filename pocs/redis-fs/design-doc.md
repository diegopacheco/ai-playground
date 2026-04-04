# redis-fs

## Overview

A CLI-based virtual filesystem that stores all files and directories in Redis. Instead of mounting via FUSE, it provides a shell-like interface where users can create files, read files, copy files, delete files, list directories, and execute bash commands against file content stored in Redis.

## Goals

- Provide a virtual filesystem CLI backed by Redis
- Support: create, read, copy, delete files
- Support: mkdir, ls, cd, pwd for directory navigation
- Support: exec to run bash commands using files from the virtual fs
- Keep it minimal with no unnecessary libraries
- Redis runs via podman

## Non-Goals

- FUSE mounting or kernel-level filesystem integration
- File permissions or ownership
- Symbolic links
- Concurrent access from multiple CLI instances

## Architecture

```
┌──────────────┐      ┌──────────────┐      ┌─────────────────┐
│   User CLI   │ ──── │  redis-fs    │ ──── │ Redis (podman)  │
│  (commands)  │      │  Rust binary │      │   port 6379     │
└──────────────┘      └──────────────┘      └─────────────────┘
```

1. **User** types commands in the redis-fs shell
2. **redis-fs binary** parses commands and translates them into Redis operations
3. **Redis** stores all file content and directory structure

## Module Structure

```
src/
  main.rs      - entry point, wires modules together
  path.rs      - path utilities (resolve, normalize, parent, basename)
  store.rs     - Redis operations (connect, get/set data, metadata, directory entries)
  commands.rs  - CLI commands (create, cat, cp, rm, mkdir, ls, exec)
  shell.rs     - REPL loop, input parsing, command dispatch
```

- **path** - pure functions for path manipulation, no Redis dependency
- **store** - thin abstraction over Redis keys, encapsulates the key naming scheme
- **commands** - each filesystem command as a function, uses path and store
- **shell** - reads input, dispatches to commands, handles TTY prompt

## CLI Commands

| Command | Description |
|---|---|
| `create <path> <content>` | Create a file with given content |
| `cat <path>` | Print file content |
| `cp <src> <dst>` | Copy a file from src to dst |
| `rm <path>` | Delete a file |
| `mkdir <path>` | Create a directory |
| `ls [path]` | List directory contents |
| `cd <path>` | Change current directory |
| `pwd` | Print current directory |
| `exec <path>` | Write file content to a temp file and execute it with bash |
| `exit` | Quit the shell |

## Data Model in Redis

| Key Pattern | Type | Purpose |
|---|---|---|
| `fs:meta:<path>` | Hash | Metadata: type (file/dir), size, created, modified |
| `fs:data:<path>` | String | File content as string |
| `fs:dir:<path>` | Set | Child entry names for a directory |

Root directory `fs:dir:/` is created automatically on first run if it does not exist.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Rust |
| Redis client | `redis` crate |
| Redis server | Redis 8.x via podman |

No FUSE dependency. No other crates beyond `redis`.

## Infrastructure Scripts

### start-redis.sh
- Starts a Redis 8.x container using podman on port 6379
- Container name: `redis-fs`
- Waits for Redis to be ready using a loop with max sleep 1

### stop-redis.sh
- Stops and removes the `redis-fs` podman container

### dump-redis.sh
- Connects to Redis on localhost:6379
- Lists all keys using `KEYS *`
- For each key, detects its type and prints the key name, type, and full content
- Handles String, Hash, Set, and List types
- Useful for debugging the virtual filesystem state

### test.sh
- Starts Redis via `start-redis.sh`
- Builds the Rust binary
- Runs a sequence of commands piped into the binary to test:
  - Create a file
  - Read the file back
  - Copy the file
  - Read the copy
  - List directory
  - Create a bash script file
  - Execute the bash script via `exec`
  - Delete files
  - Verify deletion
- Prints PASS/FAIL for each operation
- Stops Redis via `stop-redis.sh`

## Flow: Create and Read a File

1. User types `create /hello.txt Hello World`
2. redis-fs stores `"Hello World"` at `fs:data:/hello.txt`
3. redis-fs stores metadata at `fs:meta:/hello.txt` (type=file, size=11)
4. redis-fs adds `hello.txt` to the `fs:dir:/` set
5. User types `cat /hello.txt`
6. redis-fs fetches `fs:data:/hello.txt` and prints `Hello World`

## Flow: Execute a Bash Script

1. User types `create /script.sh echo "hello from redis-fs"`
2. User types `exec /script.sh`
3. redis-fs reads content from `fs:data:/script.sh`
4. redis-fs writes content to a temporary file on the real filesystem
5. redis-fs runs `bash <tempfile>` and prints the output
6. redis-fs deletes the temporary file

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Redis not running | Print clear error message and exit |
| Exec runs arbitrary code | This is a POC, user is responsible for what they execute |
| Large files in Redis | POC scope, not intended for large files |

## Success Criteria

- `start-redis.sh` starts Redis in podman
- Can create, read, copy, and delete files via the CLI
- Can list directories and navigate with cd/pwd
- Can execute bash scripts stored in the virtual fs
- `test.sh` passes all operations end to end
- `stop-redis.sh` cleanly stops Redis

# redis-fs

A CLI-based virtual filesystem that uses Redis as its storage backend. Files and directories are stored entirely in Redis, and the CLI provides a shell interface to create, read, copy, delete files, navigate directories, and execute bash scripts stored in the virtual filesystem.

## Stack

* Rust (2024 edition)
* Redis 8.x (via podman)

## How it Works

The CLI connects to Redis on `localhost:6379` and stores filesystem data using three key patterns:

| Key Pattern | Redis Type | Purpose |
|---|---|---|
| `fs:meta:<path>` | Hash | File/directory metadata |
| `fs:data:<path>` | String | File content |
| `fs:dir:<path>` | Set | Directory children |

## Commands

| Command | Description |
|---|---|
| `create <path> <content>` | Create a file with content |
| `cat <path>` | Print file content |
| `cp <src> <dst>` | Copy a file |
| `rm <path>` | Delete a file |
| `mkdir <path>` | Create a directory |
| `ls [path]` | List directory contents |
| `cd <path>` | Change directory |
| `pwd` | Print current directory |
| `exec <path>` | Execute a file as a bash script |
| `exit` | Quit |

## Running

Start Redis:
```
bash start-redis.sh
```

Build and run the CLI:
```
cargo build --release
./target/release/redis-fs
```

Stop Redis:
```
bash stop-redis.sh
```

## Testing

```
bash test.sh
```

This starts Redis, builds the binary, runs all filesystem operations, validates results, and stops Redis.

## Project Structure

```
src/
  main.rs      - entry point
  path.rs      - path utilities (resolve, normalize, parent, basename)
  store.rs     - Redis operations (connect, get/set data, metadata, directory entries)
  commands.rs  - CLI commands (create, cat, cp, rm, mkdir, ls, exec)
  shell.rs     - REPL loop, input parsing, command dispatch
```

## Debugging

Dump all Redis keys and their contents:
```
bash dump-redis.sh
```

## Result

```
❯ ./test.sh
    Finished `release` profile [optimized] target(s) in 0.03s
PASS: create file
PASS: read file
PASS: copy file
PASS: read copy
PASS: list has hello.txt
PASS: list has copy.txt
PASS: mkdir
PASS: list has scripts
PASS: create script
PASS: exec script
PASS: delete file
PASS: verify deletion
PASS: delete copy
PASS: list after deletes
PASS: cd and pwd

Results: 15 passed, 0 failed
❯ ./dump-redis.sh
=== fs:dir:/ (set) ===
scripts

=== fs:meta:/scripts/run.sh (hash) ===
type
file
size
19

=== fs:data:/scripts/run.sh (string) ===
echo redis-fs-works

=== fs:dir:/scripts (set) ===
run.sh

=== fs:meta:/ (hash) ===
type
dir
size
0

=== fs:meta:/scripts (hash) ===
type
dir
size
0
```
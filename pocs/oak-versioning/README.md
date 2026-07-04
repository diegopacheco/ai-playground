# Oak Versioning Proof of Concept

This project is a small, local-first walkthrough of [Oak](https://oak.space/), a version-control system designed for software work performed by humans and coding agents.

The repository itself is managed by Oak. It contains a tracked text file and a validation script so the core workflow can be inspected without publishing code or connecting an agent.

## What Oak is

Oak provides repositories, branches, checkpoints, diffs, history, push, pull, merge, and Git export. Its workflow differs from Git in a few important ways:

- Work happens on a feature branch rather than directly on `main`.
- Intermediate checkpoints have no messages.
- A branch has one description that becomes the squash-commit message when merged into `main`.
- Remote repositories can be mounted lazily, so file content is downloaded when read.
- Content-addressed storage and content-defined chunking reduce repeated storage and transfer, including for large binary files.

Oak does not run a coding agent and does not make AI calls. It is the version-control layer used underneath an agent or a normal terminal workflow.

## How it works

`oak init` creates local Oak metadata and places the working tree on a personal feature branch based on `main`. File edits remain in the working tree until `oak commit` creates a local checkpoint. `oak desc` records the purpose of the whole branch. `oak push` publishes local checkpoints, while `oak merge` performs a server-side squash onto `main`.

Oak stores objects by content identity. Lazy mounts fetch the repository manifest first and hydrate file contents when accessed. Changed large files are split into chunks, allowing unchanged chunks to be reused instead of transferred again.

The local cycle is:

```text
edit files -> oak status -> oak diff --print -> oak commit -> oak log
```

The remote cycle adds:

```text
oak login -> oak desc "Describe the branch outcome" -> oak push -> review -> oak merge
```

## Requirements

Oak currently supports macOS on Apple Silicon and Linux on x86_64. Lazy mounts require macOS 26 or later with the OakFS extension enabled, or FUSE 3 on Linux. A local repository does not require mount support.

## Install

Install the current CLI release:

```bash
curl -fsSL https://oak.space/install | sh
```

The installer places `oak` in `~/.local/bin`. Add that directory to `PATH` if the command is not found:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Verify the installation:

```bash
oak --version
```

Upgrade later with:

```bash
oak upgrade
```

## Run this proof of concept

From this directory:

```bash
export PATH="$HOME/.local/bin:$PATH"
test -d .oak || oak init
./test.sh
oak info
oak status
oak log --oneline
```

Change the tracked value and inspect it:

```bash
printf '%s\n' 'version=3' > message.txt
oak status
oak diff --print
oak commit
oak log --oneline
```

Restore the committed content after experimenting:

```bash
printf '%s\n' 'version=2' > message.txt
oak commit
```

## Start a new Oak repository

```bash
mkdir my-project
cd my-project
oak init
printf '%s\n' 'first file' > content.txt
oak status
oak diff --print
oak commit
oak desc "Create the initial project"
```

Authentication is only needed for server operations:

```bash
oak login
oak push --repo <organization>/<repository>
```

The explicit repository name is needed on the first non-interactive push. Later pushes can use `oak push`.

## Common commands

| Command | Purpose |
| --- | --- |
| `oak init [path]` | Create a local repository and feature branch |
| `oak clone <organization>/<repository>` | Download an Oak repository |
| `oak status` | List working-tree changes |
| `oak diff --print` | Print changes in the terminal |
| `oak commit` | Create a local, message-free checkpoint |
| `oak commit --push` | Checkpoint and publish explicitly |
| `oak log --oneline` | Show compact history |
| `oak info` | Show repository and branch metadata |
| `oak switch -c <branch>` | Create and switch to a feature branch |
| `oak desc "text"` | Set the branch description |
| `oak push` | Publish local checkpoints |
| `oak pull` | Fetch changes and merge the latest parent branch |
| `oak merge` | Squash the feature branch onto its parent on the server |
| `oak finish --desc "text"` | Describe, checkpoint, and publish a branch |
| `oak export <directory>` | Convert Oak history into a new Git repository |
| `oak mount <organization>/<repository>` | Lazily mount a remote repository |
| `oak mount list` | List active lazy mounts |
| `oak sparse set <path>` | Limit a regular clone to selected paths |

Use `oak <command> --help` for all flags. `oak --verbose <command>` prints operation timings.

## Why use Oak

Oak fits workloads that create many short-lived agent branches, checkpoint frequently, touch large repositories, or version large binary assets. The branch-level description removes repetitive checkpoint-message generation, while lazy hydration reduces startup work for large remote repositories. Each mounted task gets isolated local state rather than sharing one Git metadata directory.

Oak is less compelling for small repositories where Git is already fast, or for teams that depend heavily on the Git hosting ecosystem.

## Pros

- Fast repeated snapshots and large-file operations are explicit design goals.
- Feature branches and message-free checkpoints fit frequent agent saves.
- Lazy mounts can provide a working tree before all file contents download.
- Native chunking and deduplication support large binary files without a separate LFS setup.
- Task mounts isolate parallel work.
- `oak export` provides a path back to standard Git history.
- Oak states that it makes no AI calls and does not train on repository content.

## Cons

- Oak is beta software with a much smaller ecosystem than Git.
- Platform support is currently limited to Apple Silicon macOS and x86_64 Linux.
- Lazy mounts require one-time operating-system filesystem setup.
- Server publishing requires an Oak account and organization.
- Issues, GitHub-style review, CI, and the broad Git hosting integration ecosystem are not built in.
- Direct pushes to `main` are intentionally refused after repository creation.
- Cold initialization and process startup can be slower than Git.
- Git imports simplify some history: submodules are skipped, extra parents in octopus merges are dropped, and only the checked-out history line is converted.

## Data portability

Export the repository to Git at any time:

```bash
oak export ./git-export
cd git-export
git log --oneline
```

Keep exports outside this working tree or remove them after inspection so they are not accidentally checkpointed in Oak.

## Resources

- [Oak website](https://oak.space/)
- [Oak documentation](https://oak.space/docs)
- [Oak source repository](https://oak.space/oak/oak)
- [Oak benchmark repository](https://oak.space/oak/benchmarks)

## Oak compared with Git

- Oak uses one feature branch description; Git normally uses a message for every commit.
- Oak checkpoints are message-free; Git commits require messages.
- Oak merges feature branches into server `main` as one squash commit; Git supports several merge strategies and permits direct branch updates.
- Oak can lazily mount file contents; Git normally downloads repository objects before checkout, with partial and sparse clone options available.
- Oak chunks and deduplicates large files natively; Git commonly uses Git LFS for large binary assets.
- Oak isolates mounted tasks with separate branch state; Git worktrees share one repository metadata store.
- Oak targets agent-heavy, high-checkpoint workflows; Git is a general-purpose version-control system.
- Oak currently supports fewer platforms; Git runs across all major operating systems.
- Oak has a young hosting ecosystem; Git has mature hosting, review, CI, and integration support.
- Oak can export its history to Git; Git is already the standard interchange format for most development tooling.

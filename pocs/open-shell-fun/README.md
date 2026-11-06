# OpenShell Policy Lab

A fullstack POC (Bun backend + React frontend) on top of [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell).
It creates real OpenShell sandboxes and gives each one a **web terminal** (xterm.js over a WebSocket to a real
PTY inside the sandbox), an **Attack Lab** that plays a misbehaving agent and grades how much OpenShell blocks,
a **live policy editor** with hot reload, and a **decision log** of every ALLOWED / DENIED call.

## What is OpenShell?

[OpenShell](https://github.com/NVIDIA/OpenShell) is NVIDIA's open source (Apache-2.0) "safe, private runtime for
fleets of autonomous AI agents". Agents are most useful when they can read files, install packages, call APIs and
use credentials. OpenShell lets them do that without unrestricted access to your data, secrets or network: you
declare what each agent can touch in a **policy**, and OpenShell enforces it.

* **Sandbox**: each agent runs in an isolated container (or VM / Kubernetes pod) driven by a supervisor.
* **Kernel-level enforcement**: Landlock confines the filesystem, seccomp confines syscalls, and every outbound
  connection goes through a policy proxy (OPA at L4, HTTP method/path inspection at L7). Network is **default deny**.
* **Providers**: credentials are injected by the proxy only into requests bound for approved endpoints, so the
  agent never sees the real secret.
* **Gateway**: the control plane (CLI, SDKs for Python / TypeScript / Go / Rust, Helm chart for Kubernetes) that
  manages sandboxes, policy revisions and logs.
* **Policy prover / advisor**: formal verification flags risky new access before a policy change is approved.
* **Audit**: every decision is an OCSF event, ready to ship to a SIEM.

### Pros

* Real kernel-level isolation (Landlock + seccomp), not just a prompt-level "please don't".
* Default deny egress with fine-grained L7 rules: `read-only` lets `curl` GET `api.github.com` but blocks `POST`.
* Rules are scoped per binary and per host, and they hot reload in seconds with no sandbox restart.
* Agents never hold real credentials; the proxy injects them only for approved endpoints.
* Full audit trail (OCSF) of every allowed and denied connection and HTTP request.
* Agent agnostic: works with Claude Code, Codex, OpenCode or any process. Runs locally, on Kubernetes, and with GPUs.
* Apache-2.0, CLI plus SDKs in four languages.

### Cons

* Very young: the repo was created in February 2026 and is at 0.1.x (pre-1.0), so APIs still change between releases.
* The TypeScript SDK is published on GitHub Packages only, which needs a GitHub token to install. This POC drives the CLI instead.
* Platform limits: Linux, or macOS on Apple Silicon only; Windows via WSL 2 is experimental.
* The default sandbox image is minimal (no `curl`, no `python3`), so you usually build your own image.
  In this POC the community `base` image did not provision (`ConfigurationInvalid`), so a tiny custom image is used.
* More moving parts: a gateway service, a container runtime (Docker/Podman) and a policy schema to learn.
* L7 inspection means the proxy terminates TLS inside the sandbox.
* Anonymous telemetry is on by default (`OPENSHELL_TELEMETRY_ENABLED=false` to disable).

### Use cases

* Let coding agents (Claude Code, Codex, OpenCode) run autonomously without `--dangerously-skip-permissions` risk.
* Give an agent read-only access to GitHub, PyPI or npm while blocking writes and every other host.
* Hand agents API credentials that only work at approved endpoints, so they cannot be exfiltrated.
* Run agent fleets for teams or CI on Kubernetes with central policy and audit.
* Execute untrusted, agent-generated code safely.
* Compliance: export every agent network decision to a SIEM.

## How it works

1. The React UI calls the Bun backend under `/api` (Vite proxies it in dev).
2. The backend validates the input, then runs the `openshell` CLI with `Bun.spawn` and an argument array (never a
   shell string). User commands always come after `--`, so they can never turn into OpenShell flags.
3. The CLI talks to the local OpenShell gateway over mTLS. The gateway starts a sandbox container in Podman from
   `sandbox/Containerfile` (Ubuntu 24.04 + curl, running as uid 1000).
4. **Terminal**: xterm.js opens a WebSocket to `/api/sandboxes/:name/terminal`. The backend spawns
   `openshell sandbox exec <name> --tty -- bash -l` inside a native Bun PTY (`Bun.spawn({ terminal })`) and pipes
   bytes both ways. Keystrokes and resizes travel as small JSON messages, and output comes back as binary frames.
5. **Attack Lab**: 10 scenarios (8 attacks, 2 legit reads) run in parallel with `openshell sandbox exec`. Each
   result becomes BLOCKED or ALLOWED (non-zero exit, `HTTP 000`, `HTTP 403` or `policy_denied` count as blocked),
   and attacks are graded A / B / F.
6. **Agent console**: runs `sh -c "<command>"` inside the sandbox with `openshell sandbox exec`.
7. **Network policy**: `openshell policy update --add-endpoint host:port:access:rest:enforce --wait` adds an L7
   rule and waits until the sandbox loads the new revision. `--remove-rule` deletes it.
8. **Decision log**: `openshell logs --source sandbox` is parsed into typed OCSF events (only ALLOWED / DENIED lines)
   and shown in a TanStack Table that refreshes every 4 seconds.

## Architecture

![Architecture](printscreens/architecture.png)

## Features

* **Web terminal**: a real interactive bash shell in the sandbox in your browser, with colors, resize and reconnect.
* **Live decisions feed**: next to the terminal, every ALLOWED / DENIED call shows up within 2 seconds as you type.
* **Attack Lab**: one click runs exfiltration, metadata credential theft, raw IP egress, GitHub write, binary tampering,
  persistence, user namespace escape and password hash reading, and shows a graded scorecard with evidence.
* **Sandbox lifecycle**: create a sandbox and land straight in its terminal, delete it from the list.
* **Agent console**: run any command in the sandbox, with presets that show deny, allow, L7 deny and Landlock deny.
* **Live policy editor**: add or remove L7 network rules (binary, host, port, access preset, enforce/audit) with hot reload.
* **Effective policy view**: the exact YAML the gateway enforces, including filesystem and Landlock sections.
* **Decision log**: every ALLOWED / DENIED connection and HTTP request, with the matching rule, engine and reason.
* **Input validation**: names, hosts, ports, binaries and presets are validated before anything reaches the CLI.

## Stack

* **Bun 1.4.4**: runtime, HTTP server (`Bun.serve` routes), process spawning and test runner in one tool.
* **TypeScript 7.0.2**: strict types in the backend and frontend.
* **React 19.3.1 + Vite 8**: UI and dev server with an `/api` proxy.
* **TanStack Router 1.x**: two typed routes, `/` and `/sandboxes/$name`.
* **TanStack Query 5**: server state, polling and cache invalidation after mutations.
* **TanStack Form 1.x**: the create sandbox, console and rule forms.
* **TanStack Table 9**: the sandboxes and decision log tables.
* **xterm.js 6 + fit addon**: the browser terminal emulator.
* **Bun PTY (`Bun.spawn({ terminal })`)**: a real pseudo terminal on the host without node-pty.
* **OpenShell 0.1.7**: CLI and local gateway (Homebrew service) with Podman as the container runtime.

## REST API

| Method | Path | Body | What it does |
|---|---|---|---|
| GET | `/api/health` | | `openshell status` of the gateway |
| GET | `/api/sandboxes` | | List sandboxes `{name, phase, createdAt, policyVersion}` |
| POST | `/api/sandboxes` | `{"name":"agent-box"}` | Create a sandbox from the POC image |
| DELETE | `/api/sandboxes/:name` | | Delete a sandbox |
| POST | `/api/sandboxes/:name/exec` | `{"command":"curl -s https://api.github.com/zen"}` | Run a command, returns `{exitCode, stdout, stderr}` |
| GET | `/api/sandboxes/:name/policy` | | Effective policy `{yaml}` |
| POST | `/api/sandboxes/:name/rules` | `{"ruleName":"github_api","binary":"/usr/bin/curl","host":"api.github.com","port":443,"access":"read-only","enforcement":"enforce"}` | Add an L7 rule and wait for it to load |
| DELETE | `/api/sandboxes/:name/rules/:rule` | | Remove a network rule |
| GET | `/api/sandboxes/:name/events` | | Parsed ALLOWED / DENIED OCSF events, newest first |
| POST | `/api/sandboxes/:name/attacks` | | Runs the 10 Attack Lab scenarios, returns `{id, title, intent, layer, command, verdict, secure, exitCode, evidence}[]` |
| WS | `/api/sandboxes/:name/terminal` | client: `{"type":"input","data":"ls\r"}`, `{"type":"resize","cols":120,"rows":32}` | Interactive shell, server sends raw PTY bytes |

Errors return `{"error": "..."}`: `400` for validation, `502` when the OpenShell CLI fails.

## Key design decisions

* **CLI over SDK**: the TypeScript SDK needs a GitHub Packages token. The CLI is installed with the gateway and maps
  1:1 to the SDK calls (`create`, `exec`, `setPolicy`), so wrapping it keeps the POC dependency-free.
* **Injectable `CliRunner`**: `OpenShell` receives the runner as a function, so the tests assert the exact
  `openshell` arguments without a gateway.
* **Argument arrays only**: no shell interpolation on the host. Names must match `^[a-z0-9][a-z0-9-]*$`, so a name
  can never look like a flag (`--no-keep`), and hosts cannot contain `:`, so they cannot inject extra fields into the
  `host:port:access:protocol:enforcement` endpoint spec.
* **`exec --tty` instead of `connect` for the terminal**: `openshell sandbox connect` attaches to the sandbox main
  process, so typing `exit` moved the sandbox to `Completed`. `sandbox exec --tty -- bash -l` is a side session:
  closing the tab or typing `exit` ends only that shell and the sandbox stays `Ready`.
* **Native Bun PTY**: Bun 1.4 ships `Bun.spawn({ terminal })`, so the host side is a real TTY without native addons.
* **Honest Attack Lab labels**: each scenario names the layer that stopped it. Reading `/etc/shadow` is blocked by
  plain Unix permissions (the sandbox runs as uid 1000), not by OpenShell, and it is labelled that way.
* **`policy update` instead of `policy set`**: it changes only network rules and keeps the create-time filesystem and
  Landlock policy, which OpenShell does not allow you to change on a live sandbox.
* **`PolicyEvent` type**: `{timeMs, kind, severity, decision, target, policy, engine, reason}`, parsed from OCSF log
  lines with a regex. Plumbing events are dropped so the log shows only decisions.

## How to run

Requirements: macOS on Apple Silicon or Linux, Podman, Bun, Homebrew (macOS).

```bash
./scripts/setup.sh
./scripts/start-all.sh
./scripts/ui.sh
```

`setup.sh` installs OpenShell with the official installer if it is missing, builds the sandbox image and installs
dependencies. Tests:

```bash
./scripts/test-all.sh
```

Backend: 27 tests (OCSF log parsing, rule validation, the exact CLI arguments for each operation, Attack Lab
verdicts, terminal message parsing). Frontend: 5 tests (removable rule names from the policy YAML, scorecard grading).
Both run `tsc --noEmit`.

## Printscreens

### 1. Sandboxes

![Sandboxes](printscreens/01-sandboxes.png)

The sandbox list from the OpenShell gateway, with phase and policy version. Creating a sandbox takes you straight to
its terminal.

### 2. Terminal tab

![Web terminal](printscreens/04-web-terminal.png)

A real bash shell inside the sandbox, in the browser. `curl https://example.com` is blocked because egress is default
deny, writing to the world-writable `/var/tmp` is denied by Landlock, and `ps` shows the OpenShell supervisor as PID 1.
On the right, the live decisions feed shows the two denials the curl caused: `NET:REFUSE` on DNS and `NET:OPEN`
with `transparent_tcp_policy_denied`.

### 3. Attack Lab tab

![Attack Lab](printscreens/05-attack-lab.png)

One click runs 10 scenarios for real inside the sandbox. With no rules, all 8 attacks are blocked (grade A): network
attempts end with `HTTP 000`, binary tampering and persistence hit Landlock, `unshare` is refused by seccomp, and
`/etc/shadow` is refused by Unix permissions. The 2 legit reads (GitHub, PyPI) are blocked too, in yellow, with a tip
to add a rule.

### 4. Attack Lab with a read-only GitHub rule

![Attack Lab with rule](printscreens/06-attack-lab-with-rule.png)

After adding the `github_api` read-only rule in the Policy tab, "Read the GitHub API" flips to ALLOWED with `HTTP 200`,
while "Open an issue on GitHub" stays BLOCKED with an L7 `HTTP 403 policy_denied`. Same host, same binary: only the
HTTP method differs. Still 8 / 8 attacks blocked.

### 5. Policy tab

![Policy](printscreens/02-policy.png)

The agent console with presets on the left and the network policy editor on the right. The `github_api` rule is
active, and the POST preset shows the L7 deny body returned by the OpenShell proxy. Below is the effective policy
YAML, with the filesystem, Landlock and network sections.

### 6. Decision log tab

![Decision log](printscreens/03-decision-log.png)

Every OCSF decision for the sandbox: L7 `HTTP:POST` denied by rule `github_api`, `NET:OPEN` allowed by OPA, and the
earlier default-deny refusals with no matching rule.

## Scripts

All scripts live in `scripts/` and run from any directory of the repository.

| Script | What it does |
|---|---|
| `./scripts/setup.sh` | Installs OpenShell if missing, builds the sandbox image, installs dependencies |
| `./scripts/start-all.sh` | Starts the gateway if needed, the backend and the frontend, and prints the full link of each one |
| `./scripts/status.sh` | Shows every service port as UP or DOWN |
| `./scripts/test-all.sh` | Runs every test suite and typecheck |
| `./scripts/ui.sh` | Opens the UI in the browser |
| `./scripts/stop-all.sh` | Stops the backend and frontend (the gateway is a shared brew service and keeps running) |

Ports are declared in `scripts/ports.env`: gateway `17670`, backend `8091`, frontend `5191`.

```bash
./scripts/setup.sh
./scripts/start-all.sh
./scripts/status.sh
./scripts/ui.sh
./scripts/stop-all.sh
```

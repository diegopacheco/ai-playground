# Agent Safehouse

https://agent-safehouse.dev/

Agent Safehouse is a macOS-native sandboxing toolkit for LLM coding agents.
It uses Apple's built-in `sandbox-exec` with composable policy profiles and a deny-first model
to restrict what files and system resources an AI coding agent can access.

### How it works
* Generates a `sandbox-exec` policy file by composing modular `.sb` profile files
* Starts from `deny default` and each layer adds specific allow rules
* Wraps any command to run inside the sandbox

### run.sh
* Clones agent-safehouse from GitHub
* Generates a sandbox policy
* Runs safe commands (echo, ls) - all succeed
* Tries to read SSH private key (`~/.ssh/id_ed25519`) - **blocked** with "Operation not permitted"
* Tries to read `/etc/master.passwd` - **blocked** with "Operation not permitted"
* Reads `~/.ssh/known_hosts` - allowed because git-over-ssh needs it

## Result
```
❯ ./run.sh
=== Agent Safehouse POC ===
Policy file: /var/folders/m2/6djb07tx1hv0929qwb_znsw00000gn/T/agent-sandbox-policy.MdiiWd

[1] SAFE: Reading weather info (allowed - simple echo command)
The weather in Porto Alegre, Brazil is typically warm and humid, around 25-30C
Exit code: 0

[2] SAFE: Listing current directory (allowed - workdir is readable)
agent-safehouse
README.md
run.sh
Exit code: 0

[3] DENIED: Trying to read SSH private key (blocked by sandbox)
cat: /Users/diegopacheco/.ssh/id_ed25519: Operation not permitted
Exit code: 1

[4] DENIED: Trying to read /etc/shadow (blocked by sandbox)
cat: /etc/master.passwd: Operation not permitted
Exit code: 1

[5] SAFE: known_hosts is allowed (git-over-ssh needs it)
github.com ssh-ed25519 AAAAC3Nza...redacted...
github.com ecdsa-sha2-nistp256 AAAAE2Vjr...redacted...
Exit code: 0

=== Done ===
```

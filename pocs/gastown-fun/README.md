# GasTown

https://github.com/steveyegge/gastown

## Experience Notes

* npm instalation did not work
* I had to install via homebrew
* 

## Results



## Install

```
brew install gastown 
```

## Create HQ

```
❯ gt install ~/gt --git

🏭 Creating Gas Town HQ at /Users/diegopacheco/gt

   ✓ Created mayor/
   ✓ Created mayor/town.json
   ✓ Created mayor/rigs.json
   ✓ Created CLAUDE.md (town root identity anchor)
   ✓ Created mayor/.claude/settings.json
   ✓ Created deacon/.claude/settings.json
   ✓ Created plugins/
   ✓ Created mayor/daemon.json

   ✓ Created .gitignore
Initialized empty Git repository in /Users/diegopacheco/gt/.git/
   ✓ Initialized git repository
   ✓ Installed branch protection (auto-reverts non-main checkouts)
   ⚠ dolt not found in PATH — Dolt backend may not fully initialize
   ✓ Initialized .beads/ (town-level beads with hq- prefix)
   ✓ Provisioned 32 formulas
   ⚠ Could not create town-level agent beads: creating hq-mayor: bd create --json --id=hq-mayor --title=Mayor - global coordinator, handles cross-rig communication and escalations. --description=Mayor - global coordinator, handles cross-rig communication and escalations.

role_type: mayor
rig: null
agent_state: idle
hook_bead: null
cleanup_status: null
active_mr: null
notification_level: null --type=agent --labels=gt:agent: warning: beads.role not configured. Run 'bd init' to set.
Error: validation failed: invalid issue type: agent
   ✓ Detected overseer: diegopacheco <diego.pacheco.it@gmail.com> (via git-config)
   ✓ Created settings/escalation.json
   ✓ Created .claude/commands/ (slash commands for all agents)
   ✓ Synced 2 hook target(s)

✓ HQ created successfully!

Next steps:
  1. Add a rig: gt rig add <name> <git-url>
  2. (Optional) Configure agents: gt config agent list
  3. Enter the Mayor's office: gt mayor attach

Note: Dolt server is running (stop with gt dolt stop)
```

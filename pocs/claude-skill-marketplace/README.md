# claude-skill-marketplace

```
   ___ _                _      ____ _    _ _ _
  / __| |__ _ _  _ __| |___  / ___| | _(_) | |
 | (__| / _` | || / _` / -_) \___ \ |/ / | | |
  \___|_\__,_|\_,_\__,_\___| |___/_|\_\_|_|_|
       Skill Marketplace · local POC
```

A **native Claude Code marketplace** you can register with one command.
After `./install.sh`, it shows up in `claude plugin marketplace list` and
`/plugin install …@claude-skill-marketplace` works inside Claude Code.

---

```
❯ claude
 ▐▛███▜▌   Claude Code v2.1.143
▝▜█████▛▘  Opus 4.7 with high effort · Claude Pro
  ▘▘ ▝▝    ~/git/diegopacheco/ai-playground/pocs/claude-skill-marketplace

❯ /plugin
  ⎿  ✓ Installed hello-world. Run /reload-plugins to apply.

❯ hello-world?

⏺ You just installed the hello-world skill from the claude-skill-marketplace POC. It's a minimal placeholder skill that greets the user — purely a
  demo/test of the marketplace plugin installation flow, no real functionality.

  Want me to invoke it (/hello-world) so you can see what it outputs?

✻ Brewed for 5s
```

## Why

Claude Code already has a first-class marketplace system (`claude plugin
marketplace add <source>`). This POC is the smallest possible directory
that satisfies that spec — one marketplace, one plugin, one skill — so you
can use it as a starting point for your own private skill registry.

---

## Layout

```
claude-skill-marketplace/
├── install.sh                              registers the marketplace
├── uninstall.sh                            removes the marketplace
├── .claude-plugin/
│   └── marketplace.json                    marketplace manifest
└── plugins/
    └── hello-world/
        ├── .claude-plugin/
        │   └── plugin.json                 plugin manifest (semver)
        └── skills/
            └── hello-world/
                └── SKILL.md                the actual skill
```

`marketplace.json` lists the plugins offered. Each plugin directory is a
self-contained Claude Code plugin with its own version, and any number of
`skills/`, `commands/`, `agents/`, `hooks/` under it.

---

## Install the marketplace

```bash
./install.sh
```

Output:
```
✔ Successfully added marketplace: claude-skill-marketplace
```

Verify:
```bash
claude plugin marketplace list
```
You should see `claude-skill-marketplace` listed as a `Directory` source.

Optional scope override (default is `user`):
```bash
CSM_SCOPE=project ./install.sh
```

---

## Install the bundled plugin

From your terminal:
```bash
claude plugin install hello-world@claude-skill-marketplace
```

Or inside Claude Code:
```
/plugin install hello-world@claude-skill-marketplace
```

Verify:
```bash
claude plugin list
```
You should see `hello-world@claude-skill-marketplace` with `Status: enabled`.

Restart Claude Code so the skill registers, then the `hello-world` skill
becomes invocable.

---

## Adding your own plugin to this marketplace

1. Create a new plugin directory:
   ```
   plugins/my-plugin/
   ├── .claude-plugin/plugin.json
   └── skills/my-plugin/SKILL.md
   ```
2. Fill in `plugin.json`:
   ```json
   {
     "name": "my-plugin",
     "version": "0.1.0",
     "description": "what it does",
     "author": {
       "name": "your-name"
     }
   }
   ```
3. Add an entry in `.claude-plugin/marketplace.json`:
   ```json
   {
     "name": "my-plugin",
     "source": "./plugins/my-plugin",
     "description": "what it does",
     "version": "0.1.0"
   }
   ```
4. Validate:
   ```bash
   claude plugin validate .
   claude plugin validate ./plugins/my-plugin
   ```
5. Refresh the marketplace so Claude picks up your edits:
   ```bash
   claude plugin marketplace update claude-skill-marketplace
   ```

---

## Versioning

`version` is declared in both `plugin.json` and the matching entry in
`marketplace.json`. Bumping the version requires updating both, then:
```bash
claude plugin marketplace update claude-skill-marketplace
claude plugin update hello-world
```

Optionally, git-tag releases as `{name}--v{version}` to mark the point where
`plugin.json` and `marketplace.json` agree.

---

## Uninstall

```bash
./uninstall.sh
```

That uninstalls the `hello-world` plugin (if installed) and removes the
marketplace registration. Your local repo files are untouched.

---

## Useful commands

| Command                                                       | What it does                                |
| ------------------------------------------------------------- | ------------------------------------------- |
| `claude plugin marketplace list`                              | List all registered marketplaces            |
| `claude plugin marketplace update claude-skill-marketplace`   | Re-read this marketplace from disk          |
| `claude plugin install <name>@claude-skill-marketplace`       | Install a plugin from this marketplace      |
| `claude plugin list`                                          | Show all installed plugins and their status |
| `claude plugin uninstall <name>`                              | Uninstall a plugin                          |
| `claude plugin validate .`                                    | Validate the marketplace manifest           |
| `claude plugin validate ./plugins/<name>`                     | Validate a plugin manifest                  |

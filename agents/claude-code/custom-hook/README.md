# Custom Hooks

Custom Hooks in claude code are a way to extend the functionality of the Claude AI by allowing users to define their own hooks that can be triggered at specific points during the execution of a task. This allows for greater flexibility and customization in how Claude interacts with users and processes information.

## Installing Global Hooks

Global hooks apply to all Claude Code projects on your machine. To install a global hook:

1. Open your global settings file at `~/.claude/settings.json`
2. Add a `hooks` section with your hook configuration
3. Hooks must be in JSON format with this structure:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/your/hook-script.sh"
          }
        ]
      }
    ]
  }
}
```

Example hook script (`~/.claude/hooks/eslint-hook.sh`):

```bash
#!/bin/bash
input=$(cat)
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

if [[ -n "$file_path" && "$file_path" == *.js ]]; then
  npx eslint "$file_path"
fi
```

### Hook Configuration Levels

Claude Code supports three configuration levels:
- **Global**: `~/.claude/settings.json` - Applies to all projects
- **Project**: `.claude/settings.json` - Applies to specific project
- **Local**: `.claude/settings.local.json` - Project-specific, not committed to version control

### Hook Event Types

- **PreToolUse** - Before a tool executes
- **PostToolUse** - After a tool executes
- **UserPromptSubmit** - Validate user input
- **SessionStart/SessionEnd** - Setup and cleanup operations

### Hook Input Format

Hooks receive JSON data via stdin with this structure:

```json
{
  "session_id": "abc123",
  "tool_name": "Edit",
  "tool_input": {
    "file_path": "/path/to/file.js"
  },
  "tool_response": {
    "filePath": "/path/to/file.js",
    "success": true
  }
}
```

Access values using `jq`:
- `echo "$input" | jq -r '.tool_input.file_path'` - Path to the file being edited
- `echo "$input" | jq -r '.tool_name'` - Name of the tool being executed
- Other fields available depending on the event type

## Hook in Action

```
⏺ Update(/private/tmp/xxx/src/index.js)
  ⎿  Updated /private/tmp/xxx/src/index.js with 1 addition
       1    const apple=true;
       2    console.log(1);
       3    console.log(2);
       4 +  console.log(3);
  ⎿  PostToolUse:Edit hook returned blocking error: [/Users/diegopacheco/.claude/hooks/eslint-hook.sh]:
     /private/tmp/xxx/src/index.js
       1:7  error  'apple' is assigned a value but never used  no-unused-vars

     ✖ 1 problem (1 error, 0 warnings)
```

## Custom Hook ideas

Code Quality & Validation

1. Pre-commit linter - Run linters before allowing git commits
2. Code complexity checker - Warn when files exceed cyclomatic complexity thresholds
3. Import validator - Block imports from deprecated or forbidden packages
4. File size guard - Prevent committing files over a certain size
5. Naming convention enforcer - Validate file/function naming matches team standards

Security & Safety

6. Secret scanner - Block commits containing API keys, passwords, or tokens
7. Dependency vulnerability check - Scan for known CVEs in package.json/requirements.txt
8. HTTPS enforcer - Ensure only HTTPS URLs are used in code
9. SQL injection detector - Flag potentially unsafe database queries
10. License compliance checker - Verify all dependencies meet license requirements

Testing & CI/CD

11. Auto test runner - Run relevant tests after file edits
12. Coverage threshold guard - Block commits that drop test coverage below threshold
13. Build validator - Ensure project builds successfully before commits
14. Snapshot updater - Auto-update test snapshots when UI changes
15. E2E test trigger - Run end-to-end tests on specific file changes

Documentation

16. README sync - Update README when public API changes
17. JSDoc enforcer - Require documentation for public functions
18. Changelog generator - Auto-append entries to CHANGELOG.md
19. API documentation builder - Regenerate API docs after interface changes
20. TODO tracker - Extract and log TODOs to a tracking file

Integration & Notifications

21. Slack notifier - Post updates to team Slack channel on major changes
22. JIRA ticket linker - Ensure commits reference valid ticket numbers
23. Git branch validator - Enforce branch naming conventions
24. PR template populator - Auto-fill PR description templates
25. Code review assigner - Auto-assign reviewers based on file paths

Performance & Monitoring

26. Bundle size tracker - Log and warn about bundle size increases
27. Lighthouse runner - Run performance audits on web projects
28. Memory leak detector - Profile tests for memory issues
29. Dead code eliminator - Identify and flag unused exports

Workflow Automation

30. Environment validator - Check required environment variables exist before operations

Each hook can be implemented as a shell script that runs at specific Claude Code events, providing real-time
feedback and automation during development.
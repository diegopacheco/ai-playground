# Better Auto-Approve with Regex for Claude Code (Hook)

❯ how to apply approve-variants.py as hook for all my claude code sections?

⏺ To apply the hook globally for all Claude Code sessions:

  1. Copy the script to the hooks directory:

  mkdir -p ~/.claude/hooks
  cp approve-variants.py ~/.claude/hooks/
  chmod +x ~/.claude/hooks/approve-variants.py

  2. Add the hook configuration to ~/.claude/settings.json:

  {
    "hooks": {
      "PreToolUse": [
        {
          "matcher": "Bash",
          "hooks": [
            {
              "type": "command",
              "command": "python3 ~/.claude/hooks/approve-variants.py"
            }
          ]
        }
      ]
    }
  }

  If you already have other settings in that file, merge the hooks section into your existing config.

  3. Test it works:

  echo '{"tool_name": "Bash", "tool_input": {"command": "timeout 30 pytest"}}' | python3 ~/.claude/hooks/approve-variants.py

  Should output JSON with "permissionDecision": "allow".

  After this, Claude Code will auto-approve safe command variations like git -C /path diff, timeout 60 cargo test, or .venv/bin/pytest without prompting you each time.
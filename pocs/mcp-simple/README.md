### Test it

```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | node mcp-server.js
```

### Add into Claude Code

```bash
claude mcp add weather-server "node" "/home/diego/git/diegopacheco/ai-playground/pocs/mcp-simple/mcp-server.js"
claude mcp list
```

### Result

```bash
❯ claude
╭────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ ✻ Welcome to Claude Code!                                                                              │
│                                                                                                        │
│   /help for help, /status for your current setup                                                       │
│                                                                                                        │
│   cwd: /mnt/e35d88d4-42b9-49ea-bf29-c4c3b018d429/diego/git/diegopacheco/ai-playground/pocs/mcp-simple  │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────╯

 Tips for getting started:

 1. Run /init to create a CLAUDE.md file with instructions for Claude
 2. Use Claude to help with file analysis, editing, bash commands and git
 3. Be as specific as you would with another engineer for the best results

> Can you calculate 15 + 27 * 3 for me?

● weather-server - calculate (MCP)(expression: "15 + 27 * 3")
  ⎿  Result: 96

● 96

> I need to know the weather for New York City.

● weather-server - get_weather (MCP)(location: "New York City")
  ⎿  Weather in New York City: 72°F, partly cloudy

● 72°F, partly cloudy
```
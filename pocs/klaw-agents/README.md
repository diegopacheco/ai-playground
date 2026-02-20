# Klaw

curl -fsSL https://klaw.sh/install.sh | sh

## Result

Chat via OpenRouter:

```
export OPENROUTER_API_KEY="sk-or-xxx"
klaw chat
```

<img src="klaw-chat.png" width="600" />

Agents 

Create the cluster
```
klaw create cluster local
klaw config use-cluster local
```

Deploy the agent
```
klaw create agent lead-scorer \
    --description "Score and qualify sales leads" \
    --model claude-sonnet-4-20250514 \
    --skills crm,web-searc
```
Result:
```
❯  klaw create agent lead-scorer \
    --description "Score and qualify sales leads" \
    --model claude-sonnet-4-20250514 \
    --skills crm,web-searc
🤖 Generating AI-enhanced system prompt...
Agent 'lead-scorer' created in local/default
  Description: Score and qualify sales leads
  Model: claude-sonnet-4-20250514
  Skills: vercel-labs/agent-browser, crm, web-searc

The orchestrator will route messages to this agent based on:
  - Manual: /klaw @lead-scorer <message>
```

List Agents:
```
klaw get agents
```

Result:
```
❯ klaw get agents
Agents in local/default:

NAME         MODEL                     DESCRIPTION                    TRIGGERS
lead-scorer  claude-sonnet-4-20250514  Score and qualify sales leads
```

Describe Agent:
```
klaw get agents
```
Result:
```
❯ klaw describe agent lead-scorer
Name:        lead-scorer
Cluster:     local
Namespace:   default
Description: Score and qualify sales leads
Model:       claude-sonnet-4-20250514
Tools:       bash, read, write, edit, glob, grep
Created:     2026-02-19T21:16:31-08:00
---
System Prompt:
# lead-scorer

## Role
Score and qualify sales leads

## Guidelines
- Be helpful, accurate, and concise
- Take action when appropriate
- Ask for clarification when requirements are unclear
- Explain your reasoning when making decisions

## Skills
- **vercel-labs/agent-browser**
- **crm**
- **web-searc**

## Available Tools
- `bash`
- `read`
- `write`
- `edit`
- `glob`
- `grep`

## Best Practices
- Always verify information before acting on it
- Use appropriate tools for the task at hand
- Keep the user informed of progress on longer tasks
- Handle errors gracefully and suggest alternatives
```

Logs:
```
klaw get agents --json | jq .
```
Result:
```
❯ klaw get agents --json | jq .
[
  {
    "name": "lead-scorer",
    "cluster": "local",
    "namespace": "default",
    "description": "Score and qualify sales leads",
    "system_prompt": "# lead-scorer\n\n## Role\nScore and qualify sales leads\n\n## Guidelines\n- Be helpful, accurate, and concise\n- Take action when appropriate\n- Ask for clarification when requirements are unclear\n- Explain your reasoning when making decisions\n\n## Skills\n- **vercel-labs/agent-browser**\n- **crm**\n- **web-searc**\n\n## Available Tools\n- `bash`\n- `read`\n- `write`\n- `edit`\n- `glob`\n- `grep`\n\n## Best Practices\n- Always verify information before acting on it\n- Use appropriate tools for the task at hand\n- Keep the user informed of progress on longer tasks\n- Handle errors gracefully and suggest alternatives\n",
    "model": "claude-sonnet-4-20250514",
    "tools": [
      "bash",
      "read",
      "write",
      "edit",
      "glob",
      "grep"
    ],
    "skills": [
      "vercel-labs/agent-browser",
      "crm",
      "web-searc"
    ],
    "created_at": "2026-02-19T21:16:31.817672-08:00"
  }
]
```

Invoke the Agent:
```
klaw controller start
klaw run lead-scorer --task "Analyze new leads in HubSpot, score 1-100 based on fit"
```

After you done.
```
klaw delete cluster local
```
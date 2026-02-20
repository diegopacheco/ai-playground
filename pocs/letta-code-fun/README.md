# Letta Code

https://github.com/letta-ai/letta-code

## Experience Notes

* Login takes a long time.
* The default model on `letta` is `gpt-5-minimal`
* 

## Install

```
npm install -g @letta-ai/letta-code
```

## Result

After login (`letta`)

```
export LETTA_API_KEY="sk-let-xxx"
./run.sh
```

Result:
```
❯ letta
Failed to connect to Letta server.
Base URL: https://api.letta.com

Your credentials may be invalid or the server may be unreachable.
Let's reconfigure your setup.

   ██████    Letta Code v0.16.0
 ██      ██  GPT-5 · OAuth
 ██  ██  ██  ~/git/diegopacheco/ai-playground/pocs/letta-code-fun
 ██      ██
   ██████

● Resuming conversation with letta-code-agent
  → /agents    list all agents
  → /resume    browse all conversations
  → /new       start a new conversation
  → /init      initialize your agent's memory
  → /remember  teach your agent

● /agents
  ⎿  Already on "letta-code-agent"

● /memory
  ⎿  Memory viewer dismissed
```
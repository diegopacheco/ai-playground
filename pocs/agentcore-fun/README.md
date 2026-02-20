# Agent Core

https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-get-started-toolkit.html

## install

```
python3.13 -m venv path/to/venv
source path/to/venv/bin/activate
python3.13 -m pip install bedrock-agentcore-starter-toolkit
python3.13 --version
```

## Create an Agent

```
agentcore create
```

```
❯ agentcore create

----------------------------------------------------------------------------------------------------
🤖 AgentCore activated. Let's build your agent.
----------------------------------------------------------------------------------------------------

Where should we create your new agent?
./echoNavy

How would you like to start?
A basic starter project (recommended)

What agent framework should we use?
Strands Agents SDK

Which model provider will power your agent?
OpenAI

Add your API key now for OpenAI (optional)
********************************************************************************************************************************

What kind of memory should your agent have?
Short-term memory

Initialize a new git repository?
No

Agent initializing...
    • Template copied.
    • Venv created and installed.
✓ Agent initialized.

----------------------------------------------------------------------------------------------------
You're ready to go! Happy building 🚀
Enter your project directory using cd echoNavy
Run agentcore dev to start the dev server
Log into AWS with aws login
Launch with agentcore deploy
----------------------------------------------------------------------------------------------------
```
# openai-oauth-fun

PoC using [openai-oauth](https://github.com/EvanZhouDev/openai-oauth) to access OpenAI's API through your existing ChatGPT account via a local OAuth proxy — no paid API key needed.

## How it works

`openai-oauth` starts a local proxy server at `http://127.0.0.1:10531/v1` that authenticates requests using OAuth tokens from your local Codex/ChatGPT installation (`~/.codex/auth.json`). The proxy exposes a standard OpenAI-compatible API, so you can use the regular `openai` Node.js SDK against it.

This PoC demonstrates:
- Listing available models from your account
- Chat completion (non-streaming)
- Streaming chat completion

## Prerequisites

- Node.js 24+
- A ChatGPT account with Codex CLI authenticated (`~/.codex/auth.json` must exist)

## Stack

- Node.js 24
- TypeScript
- openai SDK
- openai-oauth (local proxy)

## How to run

```bash
./run.sh
```

This will install dependencies, build the TypeScript, start the openai-oauth proxy in the background, run the client, and then shut down the proxy.

## Output

```
❯ ./run.sh

up to date, audited 68 packages in 315ms

9 packages are looking for funding
  run `npm fund` for details

found 0 vulnerabilities

> openai-oauth-fun@1.0.0 build
> tsc


Starting openai-oauth proxy in the background...
OpenAI-compatible endpoint ready at http://127.0.0.1:10531/v1
Use this as your OpenAI base URL. No API key is required.

Available Models: gpt-5.4, gpt-5.4-mini, gpt-5.3-codex, gpt-5.2-codex, gpt-5.2, gpt-5.1-codex-max, gpt-5.1-codex, gpt-5.1, gpt-5-codex, gpt-5, gpt-5.1-codex-mini, gpt-5-codex-mini
Proxy is ready, running the client...


> openai-oauth-fun@1.0.0 start
> node dist/index.js

OpenAI OAuth Proxy PoC

=== Available Models ===
  - gpt-5.4
  - gpt-5.4-mini
  - gpt-5.3-codex
  - gpt-5.2-codex
  - gpt-5.2
  - gpt-5.1-codex-max
  - gpt-5.1-codex
  - gpt-5.1
  - gpt-5-codex
  - gpt-5
  - gpt-5.1-codex-mini
  - gpt-5-codex-mini

Using model: gpt-5.4

=== Chat Completion (gpt-5.4) ===
Response: 2 + 2 is 4.

=== Streaming Chat (gpt-5.4) ===
Response: Why do programmers prefer dark mode?

                                                                                                                Because light attracts bugs.

Done!
```

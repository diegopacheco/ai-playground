# Design Doc

Personal Agent Protocol (Poppy) draft 0.4 POC: a fictional hotel, Willow Inn, publishes itself as a
Poppy Company, and a browser Personal Agent discovers it, opens a pseudonymous Session and gets
work done through its APIs, its Company Agent and its feedback endpoint. A `poppy.json` inspector
checks any discovery document against the draft rules.

## Architecture Principles

* Follow the draft 0.4 wire format exactly where the POC implements it: `/.well-known/poppy.json`,
  `Poppy-Session` and `Poppy-Version` headers, Session JWT and delegation JWT claims, conversation
  routes, feedback body, excluded tasks document, error codes.
* Real cryptography, no fake trust: every Session JWT is ES256 signed by a delegated key, and every
  delegated key is vouched for by a delegation JWT signed by a key the Personal Agent publishes.
* Signed-out only. The draft says signed-out interactions need no OAuth server, so OAuth sign-in is
  left out on purpose and marked FUTURE.
* The Company never learns who the User is: it only sees a pairwise User ID (`sub`) that the
  Personal Agent invents per Company.
* Modular code: one module per protocol concern on both sides, in-memory state, no database.

## Features

* Discovery: `GET /.well-known/poppy.json` with organization, agent conversation endpoint,
  feedback endpoint, excluded tasks URL and a skill index entry.
* Session verification: typ, alg, kid thumbprint, delegation signature, delegation expiry,
  `iss`, `aud`, `sid`, `sub`, `htm`, `htu`, `iat` freshness and `jti` replay protection.
* Rooms API: anyone with a valid Session can search rooms; holds are keyed by the pairwise User ID,
  so a returning Personal Agent sees its own holds and nobody else's.
* Company Agent conversations: start, send, read events with a cursor, close. A rule based Company
  Agent answers about rooms, check-in, pets and parking, and points to feedback for tasks it can't do.
* Feedback: `POST {feedback.endpoint}` with `task` and `description`, length checked.
* Excluded tasks: a document listing tasks the Company won't do through Poppy, with alternatives.
* Inspector: validates a pasted `poppy.json` against the draft field rules and reports each check.
* Personal Agent UI: generates keys with WebCrypto, publishes its client metadata, signs a fresh
  Session JWT per request and shows every request on a wire log with decoded JWTs.

## Overall Diagram

```
Browser Personal Agent (React)            Willow Inn Company (Python, Granian)
  WebCrypto root key  ----PUT agent.json--> agent host (simulates agent.example)
  delegated key + delegation JWT
  GET /.well-known/poppy.json  ----------->  discovery
  Poppy-Session JWT per request ---------->  session verifier -> rooms API
                                                              -> conversations -> company agent
                                                              -> feedback
  paste poppy.json ----------------------->  inspector
```

The README renders the same flow as a hand drawn diagram.

## TradeOffs

* Pros: real ES256 signatures end to end, wire compatible field names, no database, two runtimes
  that each use their native crypto (WebCrypto in the browser, `cryptography` in Python).
* Pros: shows the core Poppy idea, controls between "blocked" and "full access", with nothing
  but a pseudonymous ID reaching the Company.
* Cons: HTTP on localhost. The draft requires HTTPS for every protocol URL; the POC relaxes that.
* Cons: the agent's client metadata is hosted by the Company process (agent host) instead of the
  agent's own domain, and carries `jwks` inline instead of a `jwks_uri`.
* Cons: no OAuth sign-in, no browser join, no handoff to a person, no streaming, no operations
  extension. All FUTURE.
* Cons: the Company Agent is keyword rules, not an LLM.

## Decisions

* Python backend over the stack's stdlib ASGI plus `cryptography`: the stdlib has no ECDSA, and
  hand rolling P-256 would be worse than one well known dependency.
* Browser WebCrypto for the Personal Agent: ECDSA P-256 signatures come out as raw `r||s`, which is
  already the JWS format, so the UI needs no crypto library.
* Direct calls with CORS instead of a Vite proxy: the Personal Agent signs `htu` with the real
  Company URL it read from `poppy.json`, and the request goes to that same URL.
* Holds keyed by `sub`: proves that a stable pairwise ID lets a Company recognize a returning User
  without knowing who they are.
* `jti` replay cache with a 120 second window, matching the `iat` freshness window.

## How to Run

```bash
./scripts/setup.sh
./scripts/start-all.sh
./scripts/ui.sh
```

## How to Test

```bash
./scripts/test-all.sh
```

## All REST Endpoints

| Method | Path | Session | Purpose |
|---|---|---|---|
| GET | `/health` | no | liveness |
| GET | `/.well-known/poppy.json` | no | Poppy discovery document |
| GET | `/.well-known/poppy-excluded-tasks.json` | no | tasks not offered through Poppy |
| PUT | `/agent-host/{name}/agent.json` | no | publish Personal Agent client metadata (simulated agent domain) |
| GET | `/agent-host/{name}/agent.json` | no | read Personal Agent client metadata |
| GET | `/api/rooms` | yes | search rooms |
| POST | `/api/holds` | yes | hold a room for this pairwise User |
| GET | `/api/holds` | yes | list this pairwise User's holds |
| POST | `/poppy/conversations` | yes | start a conversation with its first message |
| POST | `/poppy/conversations/{id}/messages` | yes | send another message |
| GET | `/poppy/conversations/{id}/events?cursor=` | yes | read events after a cursor |
| POST | `/poppy/conversations/{id}/close` | yes | close a conversation |
| POST | `/poppy/feedback` | yes | report a task that couldn't be done |
| POST | `/inspector/validate` | no | validate a `poppy.json` document |

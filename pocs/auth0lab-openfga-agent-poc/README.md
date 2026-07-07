# auth0lab-openfga-agent-poc

Python PoC for an agentic RAG authorization flow inspired by Auth0Lab and OpenFGA.

The app models a small knowledge base where an agent can retrieve candidate documents, but can only cite documents the human principal can access. The agent also needs explicit delegation before it can act for that principal.

## Why this matters

focused on identity for GenAI apps and agents. This PoC shows:

- Relationship-based access checks for agent retrieval
- Document-level filtering before content reaches an LLM boundary
- Agent delegation separated from user document access
- OpenFGA-style tuples and model files that can be moved into the OpenFGA playground

## Files

- `app.py`: local OpenFGA-style evaluator and agent retrieval flow
- `model.fga`: authorization model
- `tuples.json`: relationship tuples
- `documents.json`: protected retrieval corpus
- `tests.py`: behavior checks
- `run.sh`: runs three authorization scenarios
- `test.sh`: compiles and tests the PoC

## Run

```bash
./run.sh
```

## Test

```bash
./test.sh
```

## Expected behavior

- `user:beth` can delegate to `agent:secprep` and can cite engineering documents
- `user:carl` can delegate to `agent:secprep` and can cite only the roadmap document
- `user:dana` cannot delegate to `agent:secprep`, so the agent stops before retrieval

## OpenFGA mapping

The model follows OpenFGA's ReBAC shape:

- `folder#viewer` includes `folder#owner`
- `document#viewer` includes direct viewers, owners, and `viewer from parent`
- `document#can_cite` is derived from `viewer`
- `agent#delegate` is granted separately from document access

Docs used while shaping this:

- https://openfga.dev/docs/modeling/getting-started
- https://openfga.dev/docs/modeling/agents/rag-authorization
- https://github.com/auth0/auth0-ai-python

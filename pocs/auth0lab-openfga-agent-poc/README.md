# auth0lab-openfga-agent-poc

Python PoC for an agentic RAG authorization flow inspired by Auth0Lab and OpenFGA.

The app models a small knowledge base where an agent can retrieve candidate documents, but can only cite documents the human principal can access. The agent also needs explicit delegation before it can act for that principal.

## Reality check

This PoC is real as a local authorization teaching tool, but it is not a production Auth0, OpenFGA, or agent stack.

Real parts:

- `model.fga` defines a plausible OpenFGA-style relationship model
- `tuples.json` defines concrete user, folder, document, and agent relationships
- `app.py` enforces delegation before retrieval
- `app.py` filters documents through `can_cite` before building the answer
- Folder inheritance is implemented for document access
- `tests.py` validates the expected authorization outcomes

Simulated parts:

- No Auth0 tenant, login flow, token validation, or identity provider is used
- No OpenFGA server is called
- No Go service exists in this repository
- No real LLM is called
- No vector database or production RAG pipeline is used
- User identities are plain strings passed to the local script
- `model.fga` is not parsed or executed by the app
- Retrieval is keyword matching over `documents.json`

The main risk is model drift: `model.fga` and the Python `LocalFga` evaluator can diverge because the app does not execute the model file. The PoC proves the authorization pattern locally, not the full external integration.

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

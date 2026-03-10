# MCP2CLI

https://github.com/knowsuchagency/mcp2cli

## Install

```
uvx mcp2cli --help
```

## List tools from mcp server

```
❯ uvx mcp2cli --mcp-stdio "npx @modelcontextprotocol/server-filesystem /tmp" --list

Available tools:
  read-file                                 Read the complete contents of a file as text. DEPRECATED: Use read_tex
  read-text-file                            Read the complete contents of a file from the file system as text. Han
  read-media-file                           Read an image or audio file. Returns the base64 encoded data and MIME
  read-multiple-files                       Read the contents of multiple files simultaneously. This is more effic
  write-file                                Create a new file or completely overwrite an existing file with new co
  edit-file                                 Make line-based edits to a text file. Each edit replaces exact line se
  create-directory                          Create a new directory or ensure a directory exists. Can create multip
  list-directory                            Get a detailed listing of all files and directories in a specified pat
  list-directory-with-sizes                 Get a detailed listing of all files and directories in a specified pat
  directory-tree                            Get a recursive tree view of files and directories as a JSON structure
  move-file                                 Move or rename files and directories. Can move files between directori
  search-files                              Recursively search for files and directories matching a pattern. The p
  get-file-info                             Retrieve detailed metadata about a file or directory. Returns comprehe
  list-allowed-directories                  Returns the list of directories that this server is allowed to access.
```

## OpenAI mode

```
uvx mcp2cli --spec https://petstore3.swagger.io/api/v3/openapi.json --list
```
```
add:
  add-pet                                       POST   Add a new pet to the store.

create:
  create-user                                   POST   Create user.
  create-users-with-list-input                  POST   Creates list of users with given input array.

delete:
  delete-pet                                    DELETE Deletes a pet.
  delete-order                                  DELETE Delete purchase order by identifier.
  delete-user                                   DELETE Delete user resource.

find:
  find-pets-by-status                           GET    Finds Pets by status.
  find-pets-by-tags                             GET    Finds Pets by tags.

get:
  get-pet-by-id                                 GET    Find pet by ID.
  get-inventory                                 GET    Returns pet inventories by status.
  get-order-by-id                               GET    Find purchase order by ID.
  get-user-by-name                              GET    Get user by user name.

login:
  login-user                                    GET    Logs user into the system.

logout:
  logout-user                                   GET    Logs out current logged in user session.

place:
  place-order                                   POST   Place an order for a pet.

update:
  update-pet                                    PUT    Update an existing pet.
  update-pet-with-form                          POST   Updates a pet in the store with form data.
  update-user                                   PUT    Update user resource.
                                                                                                                                               upload:
  upload-file                                   POST   Uploads an image.
```

## Why

```
⏺ mcp2cli turns any MCP server or OpenAPI spec into a regular CLI tool at runtime, with no code generation.

  The Problem It Solves

  When an LLM connects to MCP servers, the full tool schemas (JSON) get injected into the system prompt on every single turn. With 30 tools,
  that's ~3,600 tokens per turn wasted — whether the model uses those tools or not. With multiple servers (80-120 tools), it can be 10,000+
  tokens per turn just for schemas.

  How It Works

  Instead of injecting all tool schemas upfront, mcp2cli lets the LLM discover tools on demand via CLI commands:

  1. mcp2cli --mcp <url> --list — shows a compact list of available tools (~16 tokens/tool instead of ~121)
  2. mcp2cli --mcp <url> <tool-name> --help — gets details only for the tool it needs
  3. mcp2cli --mcp <url> <tool-name> --arg value — calls the tool

  Token Savings

  ┌─────────────────────┬────────────────┬──────────────┬───────┐
  │      Scenario       │   Native MCP   │   mcp2cli    │ Saved │
  ├─────────────────────┼────────────────┼──────────────┼───────┤
  │ 30 tools, 15 turns  │ 54,525 tokens  │ 2,309 tokens │ 96%   │
  ├─────────────────────┼────────────────┼──────────────┼───────┤
  │ 120 tools, 25 turns │ 362,350 tokens │ 5,181 tokens │ 99%   │
  └─────────────────────┴────────────────┴──────────────┴───────┘

  Key Features

  - No codegen — point it at a URL and it works immediately
  - Works with any LLM — not tied to Claude's API, since it's just a CLI
  - Supports both MCP servers and OpenAPI specs
  - Caches schemas locally with configurable TTL
  - Supports auth — headers, OAuth (PKCE + client credentials)

  In short: it's a proxy that converts the "push all schemas every turn" model into a "pull what you need" model, saving massive amounts of
  context window.
```
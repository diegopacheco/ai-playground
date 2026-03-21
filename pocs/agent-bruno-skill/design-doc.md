# Design Doc: Agent Bruno Skill

## Overview

A Claude Code agent skill that scans an entire codebase, detects all HTTP/API endpoints, and generates a fully functional [Bruno](https://www.usebruno.com/) project with mapped endpoints and sample requests.

## Problem

When working on a backend project, developers need to manually create API client collections (Postman, Insomnia, Bruno) to test endpoints. This is tedious, error-prone, and gets out of sync with the actual code. This skill automates that by reading the source code and producing a ready-to-use Bruno collection.

## Goals

- Scan any backend codebase and extract all HTTP endpoints (routes, methods, paths, parameters)
- Generate a Bruno project structure with `.bru` files for every detected endpoint
- Include realistic sample request bodies, headers, query params, and path variables
- Support multiple backend frameworks/languages out of the box
- Produce a `bruno.json` collection config and organized folder structure
- Zero external dependencies — the skill is pure Claude Code orchestration

## Non-Goals

- Authentication flow automation (tokens, OAuth) — we generate placeholders instead
- Runtime validation of endpoints (no actual HTTP calls during generation)
- Supporting non-HTTP protocols (gRPC, WebSocket)
- GraphQL support

## Supported Frameworks (Detection Matrix)

| Language | Framework | Detection Strategy |
|----------|-----------|-------------------|
| Java | Spring Boot | `@GetMapping`, `@PostMapping`, `@PutMapping`, `@DeleteMapping`, `@PatchMapping`, `@RequestMapping`, `@RestController` |
| Node.js | Express | `app.get`, `app.post`, `app.put`, `app.delete`, `app.patch`, `router.get`, `router.post`, `router.put`, `router.delete` |
| Go | Gin | `router.GET`, `router.POST`, `router.PUT`, `router.DELETE`, `router.PATCH`, `gin.Default()`, `gin.New()` |
| Rust | Actix-web | `web::get()`, `web::post()`, `web::put()`, `web::delete()`, `#[get()]`, `#[post()]`, `#[put()]`, `#[delete()]`, `HttpServer::new` |
| Rust | Axum | `Router::new().route()`, `get()`, `post()`, `put()`, `delete()`, `patch()` |
| Python | Django REST | `urlpatterns`, `path()`, `re_path()`, `ViewSet`, `@api_view`, `@action` |

## Bruno Project Structure (Output)

```
bruno-collection/
  bruno.json                    # collection metadata
  environments/
    local.bru                   # localhost env variables
    dev.bru                     # dev environment (placeholder)
  {resource-group}/
    get-all-{resource}.bru      # GET /resources
    get-{resource}-by-id.bru    # GET /resources/:id
    create-{resource}.bru       # POST /resources
    update-{resource}.bru       # PUT /resources/:id
    delete-{resource}.bru       # DELETE /resources/:id
    ...
```

## .bru File Format

Each `.bru` file follows Bruno's DSL:

```
meta {
  name: Get All Users
  type: http
  seq: 1
}

get {
  url: {{base_url}}/api/users
  body: none
  auth: none
}

headers {
  Content-Type: application/json
  Accept: application/json
}

query {
  page: 1
  limit: 10
}
```

For POST/PUT requests with bodies:

```
meta {
  name: Create User
  type: http
  seq: 2
}

post {
  url: {{base_url}}/api/users
  body: json
  auth: none
}

headers {
  Content-Type: application/json
}

body:json {
  {
    "name": "John Doe",
    "email": "john@mail.com"
  }
}
```

## Skill Execution Flow

```
1. DETECT LANGUAGE & FRAMEWORK
   - Scan for build files (pom.xml, go.mod, Cargo.toml, package.json, etc.)
   - Identify the primary backend framework

2. EXTRACT ENDPOINTS
   - Search source files for route annotations/registrations
   - For each endpoint extract:
     - HTTP method (GET, POST, PUT, DELETE, PATCH)
     - URL path (with path variables)
     - Request body structure (from DTOs, structs, schemas)
     - Query parameters
     - Headers
     - Response structure (best effort)

3. INFER REQUEST SAMPLES
   - From DTO/model classes, generate realistic sample JSON bodies
   - Map path variables to sample values
   - Map query params with defaults or sensible samples

4. GENERATE BRUNO PROJECT
   - Create bruno.json with collection name derived from project name
   - Create environment files with {{base_url}} variable
   - Group endpoints by resource/controller into folders
   - Generate one .bru file per endpoint
   - Name files descriptively: {method}-{resource}-{action}.bru

5. GENERATE README
   - List all detected endpoints in a table
   - Include setup instructions for Bruno
   - Document environment variables
```

## File Layout of the Skill

```
agent-bruno-skill/
  design-doc.md
  README.md
  install.sh          # copies skill to ~/.claude/skills/
  uninstall.sh        # removes skill from ~/.claude/skills/
  skills/
    bruno-generator/
      SKILL.md        # the actual skill prompt
```

## Install / Uninstall

**install.sh** will:
- Copy the skill folder into `~/.claude/skills/bruno-generator/`
- Register in Claude Code settings if needed

**uninstall.sh** will:
- Remove `~/.claude/skills/bruno-generator/`
- Clean up any settings references

## Callouts & Risks

### 1. Framework Detection Accuracy
Not all projects follow standard patterns. Custom routing, dynamic routes, or meta-programming (reflection-based routing in Java, macros in Rust) may be missed. The skill should log what it found and what it skipped so the user knows the coverage.

### 2. Request Body Inference
Generating sample bodies from DTOs/structs is best-effort. Nested objects, generics (Java), enums, and polymorphic types add complexity. Strategy: do best-effort deep inference — follow nested types, resolve generics where possible, use enum first-values, and generate realistic sample data for all fields. Only use `"TODO"` as absolute last resort for truly unresolvable types.

### 3. Path Variable Normalization
Frameworks use different syntax: `{id}` (Spring/Go), `:id` (Express/Gin), `<id>` (Django). Bruno uses `:id` format. The skill normalizes all variations automatically:
- `{id}` -> `:id`
- `<int:id>` -> `:id`
- `:id` stays as `:id`
- `{id:\\d+}` (regex constraints) -> `:id` (strip regex)

### 4. Base URL Detection
The skill should try to detect the server port from config files (`application.properties`, `.env`, `main.go`) and set `{{base_url}}` accordingly in the environment file. Default fallback: `http://localhost:8080`.

### 5. Large Codebases
Projects with 100+ endpoints will produce many `.bru` files. The folder grouping by controller/resource keeps it navigable. The skill should also produce a summary count at the end.

### 6. Monorepo / Multi-Module
If the project has multiple services (monorepo), the skill scans ALL services and generates separate Bruno collections per service automatically. Each service gets its own `bruno-collection/` folder named after the service.

### 7. Bruno Version Compatibility
Target Bruno's current `.bru` file format (v1). The format is simple and text-based, so forward compatibility risk is low.

## Open Questions

1. **Should the skill support incremental updates?** (re-run and merge with existing Bruno project vs overwrite)
2. **Should we detect authentication schemes** (JWT, API Key, Basic) and pre-configure Bruno's auth section?
3. **Output location** — generate in `./bruno-collection/` at project root, or let user specify?

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| File format | `.bru` (Bruno DSL) | Native format, no JSON/YAML conversion needed |
| Grouping | By controller/resource | Matches how developers think about APIs |
| Sample data | Realistic defaults with deep inference | `"John Doe"` over `"string"` — best-effort on nested/generic types |
| Auth | Placeholder only | Too many auth schemes to auto-detect reliably |
| Output dir | `./bruno-collection/` | Convention over configuration, easy to find |
| GraphQL | Not supported | Out of scope — REST/HTTP only |
| Monorepo | Scan all services | Generate separate Bruno collections per service automatically |
| Frameworks | Java/Spring Boot, Node/Express, Go/Gin, Rust/Actix+Axum, Python/Django | Focused set covering most common backend stacks |
| Path variables | Normalize all to `:id` format | Bruno convention, auto-convert from `{id}`, `<id>`, regex variants |
| Validation | Test generated Bruno project | Verify `.bru` files are valid and loadable in Bruno |

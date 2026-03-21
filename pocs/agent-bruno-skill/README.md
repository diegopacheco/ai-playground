# Bruno Generator Skill

A Claude Code agent skill that scans your entire codebase, detects all HTTP/API endpoints, and generates a complete [Bruno](https://www.usebruno.com/) API client project with `.bru` files, sample requests, and environment configurations.

## Supported Frameworks

| Language | Framework |
|----------|-----------|
| Java | Spring Boot |
| Node.js | Express |
| Go | Gin |
| Rust | Actix-web, Axum |
| Python | Django REST Framework |

## What It Generates

```
bruno-collection/
  bruno.json
  environments/
    local.bru
    dev.bru
  users/
    get-all-users.bru
    get-user-by-id.bru
    create-user.bru
    update-user.bru
    delete-user.bru
  products/
    get-all-products.bru
    create-product.bru
    ...
```

Each `.bru` file contains:
- HTTP method and URL with `{{base_url}}` variable
- Headers (Content-Type, Accept)
- Sample request body with realistic data (for POST/PUT/PATCH)
- Query parameters (where applicable)
- Path variables normalized to Bruno `:param` format

## Install

```bash
./install.sh
```

Copies the skill to `~/.claude/skills/bruno-generator/`.

## Uninstall

```bash
./uninstall.sh
```

## Usage

In Claude Code, run:

```
/bruno-generator
```

The skill will:
1. Detect your project's language and framework
2. Find all HTTP endpoints in the codebase
3. Extract request bodies from DTOs/structs/models/serializers
4. Generate realistic sample data for all fields
5. Normalize path variables to Bruno `:param` format
6. Create a `bruno-collection/` folder with all `.bru` files organized by resource
7. Print a summary of all discovered endpoints

Then open the `bruno-collection/` folder in the Bruno app to start testing your API.

## Features

- **Auto-detection**: Identifies framework from build files (pom.xml, package.json, go.mod, Cargo.toml, manage.py)
- **Deep body inference**: Follows nested types, resolves generics, reads enum variants to generate realistic JSON samples
- **Path normalization**: Converts `{id}`, `<int:id>`, `<pk>` to Bruno's `:id` format automatically
- **Monorepo support**: Detects multiple services and generates separate collections for each
- **Port detection**: Reads config files to set the correct `{{base_url}}` in environment files
- **Smart grouping**: Organizes endpoints by controller/resource into folders
- **Validation**: Checks generated files for correct structure before finishing

## Monorepo Behavior

If the skill detects multiple services (multiple pom.xml, go.mod, etc.), it generates:

```
bruno-collection-service-a/
  bruno.json
  ...
bruno-collection-service-b/
  bruno.json
  ...
```

## Requirements

- [Bruno](https://www.usebruno.com/) installed on your machine to open generated collections
- Claude Code CLI

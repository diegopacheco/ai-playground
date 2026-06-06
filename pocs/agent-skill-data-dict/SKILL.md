---
name: data-dict
description: Reads a codebase and discovers its database schema wherever it is defined — Liquibase changelogs, raw SQL run through JDBC, and Hibernate/JPA entities — merges them into one model, and renders a self-contained light-theme website with three tabs (a per-table data dictionary, a hand-drawn ER diagram, and an in-browser SQL query console). Use when the user runs /data-dict or asks to document, map, visualize, or generate a data dictionary for a project's database schema.
allowed-tools: [Bash, Read, AskUserQuestion]
---

# Data Dictionary

When invoked, you read a project's source, discover every database table it defines — regardless of whether the schema lives in Liquibase changelogs, raw SQL/JDBC, or Hibernate/JPA entities — merge them into one model, and render a light-theme website that documents the whole schema.

## Global Context
- User request / scope: $ARGUMENTS — empty (scan the current repository) or a path to scan (e.g. `sample/`, `src/main`)
- Engine: `scripts/datadict.py` (zero third-party Python dependencies)
- Template: `assets/template.html`
- Output: `data-dict-report/index.html` and `data-dict-report/data.json` in the current directory

## Rules
- The schema in the report is parsed from source by the engine. Never invent tables, columns, types, sizes, or relationships, and never hand-edit `data.json`. Two runs on the same code give the same dictionary.
- Read-only against the analyzed code. The skill only writes inside `data-dict-report/`.
- The skill is generic — it makes no assumptions about any particular project. It discovers whatever schema the code defines.
- Do not add comments to any command you run.
- If the engine finds no tables, say so plainly instead of fabricating a dictionary.

## Step 1 — Pick the scan path
Decide what to scan from `$ARGUMENTS`:
- A path (e.g. `sample/`, `services/api`) → scan that path.
- Empty → scan the current repository (`.`).

The report is always written to `data-dict-report/` in the current working directory, whatever the scan path.

## Step 2 — Run the discovery engine
Invoke the engine by its installed absolute path so it finds its own template:

```bash
python3 "$HOME/.claude/skills/data-dict/scripts/datadict.py" $ARGUMENTS
```

What it does:
- Walks the scan path (skipping build/vendor directories) and parses three schema sources:
  - **Liquibase** changelogs — XML (`createTable`, `addColumn`, `addForeignKeyConstraint`, etc.), formatted-SQL, and JSON.
  - **Raw SQL / JDBC** — `CREATE TABLE` / `ALTER TABLE` in `.sql` files and in SQL strings embedded in Java (`jdbcTemplate.execute("CREATE TABLE ...")`, text blocks).
  - **Hibernate / JPA** — `@Entity` classes, reconstructing the table from `@Table`, `@Column`, `@Id`, `@GeneratedValue`, `@ManyToOne`/`@JoinColumn`, `@Enumerated`.
- Merges tables found under the same name across sources (`mixed`), resolves declared foreign keys, and infers `<name>_id → <name>` relationships where no FK is declared.
- Computes deterministic per-table facts (column count, declared sizes, estimated row width), generates SQLite DDL plus any `INSERT`s found in source for the query tab, writes `data.json`, and injects it into the template to produce `index.html`.

Notes:
- `python3` is required. If it is missing, tell the user to install Python 3.
- The engine prints a summary to stdout (tables discovered, per-source tally, relationships, report path).

## Step 3 — Open the report
```bash
open data-dict-report/index.html
```
On Linux use `xdg-open data-dict-report/index.html`.

## Step 4 — Summarize for the user
Relay the key signals from the engine's stdout:
- The project scanned and how many tables and columns were discovered.
- The per-source tally (how many tables came from Liquibase, SQL/JDBC, Hibernate, or are mixed).
- How many relationships were resolved and how many were inferred.
- The path to the rendered report.

Keep it short. The website carries the detail.

## How the website reads
- **Tables tab** — every table as a card: name, a source badge (Liquibase / SQL / JDBC / Hibernate / Mixed), origin file, description, column count and estimated row width, and a full column table (type, size, nullable, PK/FK/unique, default, description). Foreign-key columns link to the referenced table. A search box filters by table or column.
- **ER Diagram tab** — a hand-drawn, light-theme entity-relationship diagram. Boxes are color-coded by source; declared foreign keys are solid lines, inferred ones are dashed.
- **Query tab** — an in-browser SQL console (SQLite via WebAssembly) built from the discovered schema and seeded with any `INSERT`s found in source. The user can run any SQL; results, row count, and timing are shown. It runs entirely in the browser and never touches a live database.

The site is light-theme and self-contained apart from two resources loaded on demand: the SQL engine (for the query tab) and the handwriting font (for the diagram). Both degrade gracefully offline; the rest of the report works with no network.

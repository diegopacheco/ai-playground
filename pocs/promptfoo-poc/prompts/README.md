# Prompts

All prompts used by the application must be externalized in this folder,
never hardcoded in the source code.

## Rules

* One prompt per file, named after its use case, i.e. `summarize-ticket.md`.
* Prompts are templates with `$parameters` replaced at runtime.
* Parameter names are lowercase with underscores, i.e. `$user_name`, `$max_items`.
* Every parameter used in the template must be documented in the file header.
* Changing a prompt must not require changing the code.

## Format

```
---
name: summarize-ticket
params: $ticket_id, $ticket_body, $max_words
---

Summarize the ticket $ticket_id in at most $max_words words.

Ticket:
$ticket_body
```

## Loading

The application loads prompts from `prompts/`, substitutes each `$parameter`
with a runtime value and fails loud when a parameter has no value.

---
name: agentsfile-lint
description: Lint every Agentsfile in the repository with agentsfile-lint and fix the errors it reports.
---

Run `cargo run -q --manifest-path linter/Cargo.toml -- Agentsfile spec/fixtures/valid/*.agentsfile`.
Fix every `error[...]` line. Treat `warning[...]` lines as suggestions.

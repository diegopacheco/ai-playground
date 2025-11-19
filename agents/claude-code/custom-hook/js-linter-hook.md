---
on_edit:
  pattern: "**/*.js"
  command: |
    npx eslint "${file_path}"
---

# JS Linter Hook

Runs ESLint on JavaScript files whenever they are edited.

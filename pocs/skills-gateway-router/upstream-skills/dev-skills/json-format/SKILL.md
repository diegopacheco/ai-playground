---
name: json-format
description: Validates a JSON file and pretty prints it with two space indentation, or reports where the JSON is invalid.
---

Request: $ARGUMENTS

Run `node ${SKILL_DIR}/json-format.js <path>`.
If it exits 0, show the formatted JSON. Write it back to the file only when the user asked for that.
If it exits 1, answer with the error line it printed and do not touch the file.

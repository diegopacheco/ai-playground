---
name: write-tests
description: Write unit tests for the core logic of a file using the test framework the project already uses. Use when the user asks to add tests or runs /code:write-tests.
---

# Write Tests

Write tests for the target given in the arguments: `$ARGUMENTS`.

1. Detect the test framework the project already uses. Do not add a new one.
2. List the behaviors of the target that carry business meaning: edge cases, boundaries, error paths.
3. Write one test per behavior. Each test name says why the behavior matters.
4. Do not test getters, the language or the framework.
5. Run the tests and show the output.

Every test must be able to fail when the logic it covers changes.

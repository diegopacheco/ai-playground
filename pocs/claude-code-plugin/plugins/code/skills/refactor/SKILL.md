---
name: refactor
description: Refactor a file or function to be simpler without changing behavior. Use when the user asks to refactor, simplify or clean up code or runs /code:refactor.
---

# Refactor

Refactor the target given in the arguments: `$ARGUMENTS`.

1. Read the target and every caller of the code you will change.
2. Find the tests that cover it and run them first. When none exist, say so before changing anything.
3. Apply only behavior preserving changes: remove duplication, inline single use abstractions, rename unclear names, flatten nested conditionals, delete dead code.
4. Run the same tests again and show that they still pass.

Report what changed in at most five bullets and the test result before and after.

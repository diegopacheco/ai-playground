---
name: trace
description: Trace a value or a request through the code from its entry point to where it goes wrong. Use when the user asks where a value comes from, why it is wrong, or runs /debug:trace.
---

# Trace

Trace what is described in the arguments: `$ARGUMENTS`.

1. Find the entry point where the value or request enters the code.
2. Follow every call and assignment that touches it, in order.
3. At each hop, write the file and line and what the value is at that point.
4. Stop at the first hop where the value stops matching what is expected.

Report the path as a numbered list of hops and name the hop where it goes wrong and why.

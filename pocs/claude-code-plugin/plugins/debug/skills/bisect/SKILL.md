---
name: bisect
description: Find the commit that introduced a regression with git bisect and a check command. Use when something used to work and now does not, or when the user runs /debug:bisect.
---

# Bisect

Find the commit that introduced the regression described in the arguments: `$ARGUMENTS`.

1. Ask for or infer a known good commit and a known bad commit. Default bad is `HEAD`.
2. Write a check command that exits 0 on good and non zero on bad.
3. Run `git bisect start <bad> <good>` and `git bisect run <check>`.
4. Always finish with `git bisect reset`, even when the run fails.

Report the first bad commit hash, its message, and the lines of its diff that explain the regression.

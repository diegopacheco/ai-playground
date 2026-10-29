---
name: env-check
description: Check that the tools and runtimes a project needs are installed with compatible versions. Use when a build fails on a new machine, a command is not found, or the user runs /troubleshoot:env-check.
---

# Env Check

Check the environment for the project in the current directory, or for the tools listed in the arguments: `$ARGUMENTS`.

1. Read the build files to find the required tools and versions: `package.json`, `pom.xml`, `build.gradle`, `Cargo.toml`, `go.mod`, `pyproject.toml`, `.tool-versions`, `.nvmrc`.
2. For each tool run its version command.
3. Compare the installed version with the required one.

Never print environment variable values. Report a table with the tool, the required version, the installed version and OK, MISSING or MISMATCH.

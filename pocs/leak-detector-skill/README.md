# Leak Detector Skill

A Claude Code and Codex skill that scans your codebase for leaked PII, secrets/credentials, and security vulnerabilities before you push.

## Result

```
❯ /leak-detector

⏺ I'll scan the codebase for PII, secrets, and security vulnerabilities. Let me start by discovering all source files.

⏺ Searched for 9 patterns, read 1 file (ctrl+o to expand)

⏺ Found 6 files. Let me read them and run the security scans.

⏺ Searched for 1 pattern, read 6 files (ctrl+o to expand)

⏺ Now let me run the PII, secrets, and security disaster scans in parallel.

⏺ Searched for 15 patterns (ctrl+o to expand)

⏺ All matches are from documentation files (SKILL.md, design-doc.md) referencing patterns as examples — not actual PII, secrets, or
vulnerabilities in code. Let me also check for .env files and .gitignore.

⏺ Searched for 2 patterns (ctrl+o to expand)

⏺ No .env files found. Scan complete. Here's the report:

---
Scan complete: 6 files scanned

  PII:              0 findings
  Secrets:          0 findings
  Security Issues:  0 findings
  --------------------------
  Total:            0 findings

Verdict: CLEAN - no leaks detected
```

## What It Detects

### PII
Emails, Social Security Numbers, credit card numbers, phone numbers.

### Secrets and Credentials
AWS keys, GitHub tokens, Google API keys, Slack tokens, OpenAI/Stripe keys, private keys, hardcoded passwords, connection strings with embedded credentials, `.env` files not in `.gitignore`.

### Security Disasters
SQL injection, command injection, hardcoded admin credentials, debug mode in prod, open CORS, disabled TLS/SSL, insecure deserialization, weak crypto, exposed ports, secrets in logs, open redirects, path traversal.

## Install

```bash
./install.sh
```

Copies the skill to `~/.claude/skills/leak-detector/` and `~/.codex/skills/leak-detector/`.

## Uninstall

```bash
./uninstall.sh
```

## Usage

Inside Claude Code or Codex:

```
/leak-detect
```

## Output

```
[PII]      Email found in src/config.go:42
           -> diego****@****.com
           -> Move to environment variable or secrets manager

[SECRET]   AWS Access Key found in deploy.sh:17
           -> AKIA****XXXX
           -> Rotate immediately, use IAM roles or env vars

[DISASTER] SQL Injection found in api/users.go:89
           -> query := "SELECT * FROM users WHERE id=" + userId
           -> Use parameterized queries

Scan complete: 128 files scanned

  PII:              1 findings
  Secrets:          1 findings
  Security Issues:  1 findings
  --------------------------
  Total:            3 findings

Verdict: DO NOT PUSH - critical issues found
```

## Verdicts

| Condition | Verdict |
|---|---|
| Any SECRET or DISASTER found | `DO NOT PUSH - critical issues found` |
| Only PII found | `REVIEW BEFORE PUSH - PII detected` |
| Nothing found | `CLEAN - no leaks detected` |

## Project Structure

```
leak-detector-skill/
  README.md
  design-doc.md
  install.sh
  uninstall.sh
  skills/leak-detector/
    SKILL.md
```

## Requirements

- Claude Code or Codex installed
- No external dependencies

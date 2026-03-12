# Leak Detector Skill

A Claude Code and Codex skill that scans your codebase for leaked PII, secrets/credentials, and security vulnerabilities before you push.

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

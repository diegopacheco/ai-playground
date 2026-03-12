# Leak Detector Skill - Design Doc

## Problem

Developers accidentally commit sensitive data and dangerous code into repositories.
This includes PII, secrets, credentials, and security vulnerabilities.
Once pushed, these leaks are hard to remediate and can lead to breaches, fines, and hacks.

## Goal

Build a Claude Code and Codex skill (`/leak-detect`) that scans code for three categories
of leaked or dangerous content, reports findings with file locations, and suggests remediation.

## Three Detection Pillars

### 1. PII (Personally Identifiable Information)

Detects personal data that should never be in source code.

| What | Pattern |
|---|---|
| Email addresses | `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` |
| Social Security Numbers | `\b\d{3}-\d{2}-\d{4}\b` |
| Phone numbers | `\b\d{3}[-.]?\d{3}[-.]?\d{4}\b` and international formats |
| Credit card numbers | `\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b` |
| IP addresses (hardcoded) | `\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b` near config or assignment |
| Physical addresses | Street addresses, zip codes near person-related context |
| Names in data files | JSON/CSV/YAML with fields like `name`, `first_name`, `last_name` containing real-looking values |

### 2. Secrets and Credentials

Detects secrets that answer: "Is it safe to commit this file?"

| What | Pattern |
|---|---|
| AWS Access Key | `AKIA[0-9A-Z]{16}` |
| AWS Secret Key | Strings near `aws_secret_access_key` labels |
| GitHub Token | `ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_` prefixes |
| Google API Key | `AIza[0-9A-Za-z\\-_]{35}` |
| Slack Token | `xox[bpors]-[a-zA-Z0-9-]+` |
| Generic API Key | `api[_-]?key\s*[:=]\s*['"][a-zA-Z0-9]{16,}['"]` |
| Generic Secret | `secret\s*[:=]\s*['"][a-zA-Z0-9]{16,}['"]` |
| Generic Password | `password\s*[:=]\s*['"][^'"]{8,}['"]` |
| Generic Token | `token\s*[:=]\s*['"][a-zA-Z0-9]{16,}['"]` |
| Private Keys | `-----BEGIN (RSA|EC|DSA|OPENSSH) PRIVATE KEY-----` |
| Connection Strings | URIs with embedded credentials `protocol://user:pass@host` |
| .env files tracked | `.env` files not in `.gitignore` |
| Hardcoded DB credentials | `jdbc:`, `mongodb://`, `redis://` with inline passwords |

### 3. Security Disasters

Detects vulnerabilities that answer: "Would I be hacked if I push this to prod?"

| What | What to look for |
|---|---|
| SQL Injection | String concatenation in SQL queries instead of parameterized queries |
| Command Injection | User input passed to `exec`, `system`, `os.popen`, `subprocess` without sanitization |
| Hardcoded Admin Credentials | `admin/admin`, `root/root`, `password123`, default credentials in code |
| Debug Mode in Prod Config | `DEBUG=True`, `debug: true` in production config files |
| CORS Wide Open | `Access-Control-Allow-Origin: *` or `cors({ origin: '*' })` |
| Auth Disabled | Comments or flags like `auth: false`, `skipAuth`, `noAuth`, `@PermitAll` on sensitive endpoints |
| TLS/SSL Disabled | `verify=False`, `rejectUnauthorized: false`, `InsecureSkipVerify: true` |
| Exposed Stack Traces | Error handlers returning full stack traces to clients |
| Insecure Deserialization | `pickle.loads`, `yaml.load` (without safe loader), `eval()` on user data |
| Weak Crypto | `MD5`, `SHA1` used for passwords or security, `DES`, `RC4` usage |
| Open Redirect | Redirects using unvalidated user-supplied URLs |
| Path Traversal | File operations using unsanitized user input (`../` not blocked) |
| Exposed Ports/Services | Docker/k8s configs exposing internal services to `0.0.0.0` or public |
| No Rate Limiting | Auth endpoints, APIs without any rate limiting config |
| Secrets in Logs | Logging statements that print tokens, passwords, or keys |

## Trigger

- On-demand via `/leak-detect`
- Scans all files in the current working directory (respecting .gitignore)

## Excluded Files

The skill skips:

- Binary files (images, compiled artifacts, archives)
- Lock files (`package-lock.json`, `yarn.lock`, `go.sum`, `Cargo.lock`)
- Vendored dependencies (`node_modules/`, `vendor/`, `target/`)
- `.git/` directory
- Files matching `.gitignore` patterns
- Test fixture files with fake/mock data (best effort)

## Output Format

For each finding:

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
```

Summary:

```
Scan complete: 128 files scanned

  PII:              3 findings
  Secrets:          2 findings
  Security Issues:  5 findings
  --------------------------
  Total:           10 findings

Verdict: DO NOT PUSH - critical issues found
```

Verdict logic:
- Any SECRET or DISASTER finding -> `DO NOT PUSH - critical issues found`
- Only PII findings -> `REVIEW BEFORE PUSH - PII detected`
- No findings -> `CLEAN - no leaks detected`

## Architecture

```
/leak-detect (skill invocation)
    |
    v
[1] Glob all files in working directory
    |
    v
[2] Filter out excluded files (binaries, lock files, vendor, .git)
    |
    v
[3] For each file:
    - Grep for PII patterns
    - Grep for secret/credential patterns
    - Read suspicious files and analyze for security disasters
    |
    v
[4] Collect findings with: category, file, line, masked snippet, suggestion
    |
    v
[5] Output report with verdict
```

## Skill Structure

```
skills/leak-detector/
  SKILL.md
```

Single file skill. All detection logic lives in the SKILL.md prompt.
No external dependencies. No libraries.
Uses Claude Code built-in tools (Glob, Grep, Read) to perform the scan.
Security disaster detection relies on Claude reasoning over code patterns, not just regex.

## Compatibility

- Claude Code: invoked as `/leak-detect`
- Codex: same SKILL.md loaded as a skill/prompt

## Limitations

- Regex-based PII and secret detection has false positives
- Security disaster detection depends on Claude reasoning quality
- Does not scan git history (only current working tree)
- Does not block commits (reporting only)
- Cannot detect secrets in encrypted or obfuscated form
- Large repos may take longer to scan

## Future Work

- Hook into pre-commit to scan staged files automatically
- Custom pattern support via `.leak-detect.yml` config file
- Severity levels (critical, high, medium, low)
- Allow-list for known false positives
- Git history scanning mode
- CI/CD integration mode

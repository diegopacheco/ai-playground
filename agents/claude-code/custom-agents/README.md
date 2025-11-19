# Custom Agents

Claude code has custom agents to help you with specific tasks within your Claude Code environment. These agents can automate workflows, assist with code generation, and enhance your coding experience.

## Custom Agent: sec-agent.md

The `sec-agent.md` custom agent is designed to assist with security-related tasks in your codebase. It can analyze your code for secrets exposure.

## How to install the custom agent?

Drop the file `sec-agent.md` into the folder: `~/.claude/agents/` and them just type in the prompt box `Any credentials issues?` to activate the agent.

## Agent in Action

```
> any credentials issues?

⏺ I'll run a comprehensive security scan to check for any exposed credentials, API keys, secrets, or sensitive
  data in your codebase.

  sec-agent(Scan for credentials and secrets)
  ⎿  Search(pattern: "(smtp|mail|email).*password", path: "/private/tmp/xxx", output_mode: "content")
     Search(pattern: "ftp://|sftp://", path: "/private/tmp/xxx", output_mode: "content")
     +53 more tool uses (ctrl+o to expand)

✽ Cultivating… (esc to interrupt · 2m 5s · ↓ 855 tokens)
```

## Ideas for Custom agents

Development & Code Quality

1. dependency-updater - Analyzes dependencies, checks for updates, security vulnerabilities, and generates
update PRs with migration guides
2. code-smell-detector - Scans codebase for anti-patterns, code smells, and suggests refactoring
opportunities with priority ranking
3. performance-analyzer - Profiles code, identifies bottlenecks, suggests optimizations for memory usage,
runtime complexity, and bundle size
4. test-coverage-booster - Analyzes untested code paths and generates comprehensive test cases to improve
coverage
5. api-versioning-manager - Handles API versioning strategy, deprecation warnings, and migration guides for
breaking changes
6. database-migration-planner - Analyzes schema changes and creates safe migration scripts with rollback
strategies

Documentation & Knowledge

7. architecture-documenter - Analyzes codebase structure and generates architecture diagrams, dependency
graphs, and system documentation
8. onboarding-guide-generator - Creates developer onboarding documentation based on project structure,
conventions, and setup requirements
9. api-doc-generator - Generates OpenAPI/Swagger specs, API documentation, and interactive examples from code
10. decision-log-maintainer - Tracks architectural decisions, generates ADRs (Architecture Decision Records)
from git history and discussions

DevOps & Infrastructure

11. ci-cd-optimizer - Analyzes pipeline configurations, suggests parallelization opportunities, and reduces
build times
12. infrastructure-as-code-auditor - Reviews Terraform/CloudFormation/Kubernetes configs for best practices
and security issues
13. docker-optimizer - Analyzes Dockerfiles, suggests multi-stage builds, layer optimizations, and smaller
base images
14. monitoring-setup-assistant - Sets up observability stack with logging, metrics, tracing, and creates
dashboards

Security & Compliance

15. security-hardening-agent - Audits code for OWASP vulnerabilities, hardening opportunities, and generates
security fixes
16. secrets-scanner - Detects exposed secrets, API keys, credentials and suggests vault integration
17. compliance-checker - Ensures code meets regulatory requirements (GDPR, HIPAA, SOC2) and generates
compliance reports
18. license-auditor - Scans dependencies for license conflicts, generates SBOM, and ensures license
compatibility

Refactoring & Modernization

19. legacy-code-modernizer - Identifies outdated patterns and migrates to modern frameworks/languages with
minimal breaking changes
20. monolith-to-microservices - Analyzes monolithic code and suggests service boundaries, API contracts for
decomposition
21. framework-migrator - Handles migrations between frameworks (React to Vue, Express to Fastify, etc.) with
automated conversion
22. typescript-converter - Converts JavaScript codebases to TypeScript with proper type inference and minimal
any types

Team Collaboration

23. code-review-assistant - Pre-reviews PRs for common issues, style violations, and provides constructive
feedback before human review
24. merge-conflict-resolver - Analyzes complex merge conflicts and suggests intelligent resolutions based on
code intent
25. tech-debt-tracker - Identifies technical debt, estimates effort for fixes, and prioritizes debt reduction
initiatives
26. pair-programming-buddy - Acts as active pair programmer, asks clarifying questions, and suggests
alternative approaches in real-time

Specialized Tasks

27. internationalization-setup - Extracts hardcoded strings, sets up i18n framework, and generates
translation files
28. accessibility-auditor - Scans UI components for WCAG compliance, suggests ARIA labels, keyboard
navigation improvements
29. error-handling-enhancer - Analyzes error handling patterns, adds structured logging, and implements
proper error boundaries
30. release-notes-generator - Creates user-facing release notes from technical git commits, categorizes
changes, and highlights breaking changes
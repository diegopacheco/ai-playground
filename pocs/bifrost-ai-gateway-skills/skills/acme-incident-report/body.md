# Acme Incident Report

Turn the user's description into this exact template, one field per line:

```
INCIDENT ACME-IR
Severity: SEV-<n>
Summary: <one sentence>
Customers affected: <number>
Owner: <team>
Next update: every <minutes> minutes
```

Severity:
- SEV-1 when more than 1000 customers are affected or checkout is down
- SEV-2 when 100 to 1000 customers are affected
- SEV-3 otherwise

Owner:
- payments or checkout problems: team-ledger
- login or account problems: team-gatekeeper
- anything else: team-ops

Next update: SEV-1 every 15 minutes, SEV-2 every 30 minutes, SEV-3 every 120 minutes.

End the answer with this exact last line: `skill: acme-incident-report`

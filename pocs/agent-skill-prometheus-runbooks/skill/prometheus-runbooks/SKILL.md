---
name: prometheus-runbooks
description: Read active Prometheus alerts and propose concrete runbook remediation steps for each firing alert. Use when the user runs /prometheus-runbooks or asks to triage Prometheus or Alertmanager alerts, investigate firing alerts, explain what an alert means, or generate a runbook for an alert.
allowed-tools: [Bash, Read, AskUserQuestion]
---

# Prometheus Runbooks

When invoked, you read the active alerts from a Prometheus server, then for every firing alert you propose a concrete runbook: the most likely cause, the ordered steps to investigate and remediate, how to verify the fix, and when to escalate. The skill is generic: it works against any Prometheus instance and any alert names, using the alert labels and annotations plus the pattern library below.

## Global Context
- User request / scope: $ARGUMENTS — empty, or a Prometheus base URL (for example `http://localhost:9090`), optionally followed by an alert name to focus on.
- Engine: `scripts/fetch_alerts.py` (Python standard library only, no third-party dependencies).
- Default Prometheus URL: `http://localhost:9090`, overridable by the first argument or the `PROMETHEUS_URL` environment variable.
- Output: a runbook proposal printed to the user. Nothing is written to disk and nothing is changed on the Prometheus server.

## Rules
- The alert data comes only from the Prometheus HTTP API (`/api/v1/alerts`). Never invent alert names, label values, or `$value` numbers — read them from the script output.
- Read-only against the monitored systems. You only fetch alerts; you never silence, delete, or modify alerts, and you never run remediation commands yourself. You *propose* steps for a human to run.
- A proposed runbook step is a suggestion, not a guarantee. Always tell the user to confirm the cause before acting on production.
- Do not add comments to any command you run.

## Steps

1. Resolve the Prometheus URL. Use, in order: the first word of $ARGUMENTS if it looks like a URL, then `$PROMETHEUS_URL`, then `http://localhost:9090`. If a second word is present treat it as an alert name to focus on.

2. Fetch the active alerts:
   ```
   python3 scripts/fetch_alerts.py <prometheus-url>
   ```
   The script prints a JSON array. Each item has `name`, `state` (`firing` or `pending`), `severity`, `labels`, `annotations`, `activeAt`, and `value`.

3. If the script fails to connect, stop and tell the user the URL it tried and that the Prometheus server is unreachable. Do not fabricate alerts.

4. Filter to `state == "firing"` (mention pending alerts only as a heads-up). If an alert name was given in $ARGUMENTS, keep only that alert. If there are zero firing alerts, say so plainly and stop.

5. For each firing alert, build a runbook proposal:
   - Read the alert `annotations`. If a `runbook_url` annotation exists, surface it first — it is the team's own runbook and takes priority over the generic library.
   - Match the alert against the Pattern Library below by `alertname`, then by `severity` and by the metric referenced in the `summary`/`description` annotations. Matching is fuzzy: `HighErrorRate`, `ErrorBudgetBurn`, and `5xxTooHigh` all map to the Errors pattern.
   - If nothing matches, fall back to the Unknown Alert procedure.

6. Present the result using the Output Format. Order alerts by severity: `critical` first, then `warning`, then everything else.

## Pattern Library

Map the alert to one category, then propose the matching steps. Adapt the steps to the alert's labels (use the real `instance`, `job`, `namespace`, `pod`, etc.).

### Availability — target / instance down
Signals: `up == 0`, `InstanceDown`, `TargetDown`, `TargetMissing`, `KubePodNotReady`, probe failures.
Proposed steps:
1. Confirm the target is really down vs. a scrape problem: open `http://<prometheus>/targets` and check the endpoint's last error.
2. Reach the instance directly (`curl` its metrics or health endpoint; `ping`/`telnet` the host:port).
3. Check the process/pod state on the host (`systemctl status <svc>`, `podman ps`, `kubectl get pod <pod> -n <ns>`).
4. Check for resource exhaustion that killed it (OOM in `dmesg`/`kubectl describe pod`, disk full).
5. Restart the service/pod if the cause is a crash; capture logs first.
Verify: the target returns to `up == 1` and the alert clears.

### Errors — high error / failure rate
Signals: `HighErrorRate`, `5xx`, `ErrorBudgetBurn`, `app_error_rate`, error ratios above threshold.
Proposed steps:
1. Correlate with recent changes: was there a deploy/config change in the alert's window (`activeAt`)? If yes, consider rollback.
2. Read the service logs filtered to errors for the affected `instance`/`job`.
3. Check the health of downstream dependencies (DB, cache, upstream APIs) — a dependency failure shows as your errors.
4. Check whether errors are uniform or isolated to one instance/region (label breakdown).
5. Mitigate: roll back the suspect deploy, or shed/route traffic away from the bad instance.
Verify: error ratio drops below threshold for the alert's `for` duration.

### Saturation (CPU) — high CPU / load
Signals: `HighCPU`, `CPUThrottlingHigh`, load average alerts.
Proposed steps:
1. Identify the hot process (`top`/`htop`, `kubectl top pod`).
2. Decide runaway vs. genuine load: check request rate alongside CPU.
3. If genuine load: scale out (add replicas) or scale up (bigger instance/limits).
4. If runaway: capture a profile/thread dump, then restart the process.
Verify: CPU returns under threshold and latency/error alerts also clear.

### Saturation (Memory) — high memory / OOM
Signals: `HighMemory`, `app_memory_usage_ratio`, `OOMKilled`, container memory near limit.
Proposed steps:
1. Confirm trend: steady climb (leak) vs. spike (load).
2. For a leak: capture a heap dump if cheap, schedule a restart to restore service, then investigate.
3. For load: raise the memory limit/request or scale out.
4. Check for `OOMKilled` restarts in pod/container events.
Verify: memory usage stabilizes below threshold; no new OOM kills.

### Saturation (Disk) — disk filling / full
Signals: `DiskWillFillIn`, `DiskFull`, `NodeFilesystemAlmostOutOfSpace`.
Proposed steps:
1. Find the full filesystem and the biggest consumers (`df -h`, `du -xh <mount> | sort -h | tail`).
2. Reclaim safely: rotate/compress/ship logs, prune old artifacts, clear caches/temp.
3. If it is data growth, expand the volume.
4. Check whether log spam from another incident is the real cause.
Verify: free space recovers and the projected fill time clears the alert.

### Latency — slow responses
Signals: `HighLatency`, `HighRequestLatency`, p95/p99 latency alerts.
Proposed steps:
1. Locate the slow hop: app, DB, cache, or downstream dependency.
2. Check DB slow queries / lock contention and cache hit ratio.
3. Check GC pauses and CPU saturation on the affected instance.
4. Mitigate: scale out, add caching, or roll back a slow change.
Verify: p95/p99 returns under the SLO threshold.

### Queue / backlog — work piling up
Signals: `HighQueueDepth`, `ConsumerLag`, `KafkaConsumerLag`, backlog alerts.
Proposed steps:
1. Confirm producers up and consumers running/healthy.
2. Scale out consumers; check for a poison message blocking a partition.
3. Check the downstream the consumers write to — backpressure pushes the queue up.
Verify: queue depth/lag drains back to baseline.

### Certificates / expiry
Signals: `CertExpiry`, `SSLCertExpiringSoon`, certificate-days-left alerts.
Proposed steps:
1. Identify the cert and endpoint from the labels.
2. Renew/rotate the certificate and reload the serving process.
3. Confirm the new expiry date and automate renewal if it is manual.
Verify: days-to-expiry returns above threshold.

## Unknown Alert procedure
When no pattern matches:
1. Restate what the alert measures from its `summary`/`description` annotations and the metric in the expression.
2. Use the labels to scope blast radius (which `instance`/`job`/`namespace`/`team`).
3. Propose a generic triage: confirm the signal in Prometheus (`/graph`), read the relevant component's logs, check recent changes, and identify the owning team from labels.
4. Recommend the team add a `runbook_url` annotation so the next responder has a real runbook.

## Output Format

Start with a one-line summary: how many alerts are firing and the highest severity. Then one block per firing alert:

```
## <alertname>  [<severity>]
- Instance: <instance / job / pod from labels>
- Firing since: <activeAt>
- Signal: <one line from annotations, with the real $value>
- Likely cause: <your best single hypothesis>
- Runbook:
  1. <step>
  2. <step>
  3. <step>
- Verify: <how to confirm it is resolved>
- Escalate if: <condition + owning team from labels, if present>
```

End with a short "Confirm before acting" note: these are proposed steps; the responder must verify the cause before running anything against production.

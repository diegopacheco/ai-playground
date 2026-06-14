import json
import os
import sys
import urllib.error
import urllib.request


def resolve_url(argv):
    if len(argv) > 1 and argv[1].startswith("http"):
        return argv[1].rstrip("/")
    env = os.environ.get("PROMETHEUS_URL")
    if env:
        return env.rstrip("/")
    return "http://localhost:9090"


def fetch(base):
    url = base + "/api/v1/alerts"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


def normalize(payload):
    alerts = payload.get("data", {}).get("alerts", [])
    rows = []
    for a in alerts:
        labels = a.get("labels", {})
        annotations = a.get("annotations", {})
        rows.append(
            {
                "name": labels.get("alertname", "unknown"),
                "state": a.get("state"),
                "severity": labels.get("severity", "none"),
                "instance": labels.get("instance")
                or labels.get("pod")
                or labels.get("job")
                or "unknown",
                "labels": labels,
                "annotations": annotations,
                "runbook_url": annotations.get("runbook_url"),
                "activeAt": a.get("activeAt"),
                "value": a.get("value"),
            }
        )
    rows.sort(key=lambda r: (r["state"] != "firing", r["name"]))
    return rows


def main():
    base = resolve_url(sys.argv)
    try:
        payload = fetch(base)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        sys.stderr.write(
            "ERROR: could not reach Prometheus at " + base + " (" + str(exc) + ")\n"
        )
        sys.exit(1)
    if payload.get("status") != "success":
        sys.stderr.write("ERROR: Prometheus returned status " + str(payload.get("status")) + "\n")
        sys.exit(1)
    print(json.dumps(normalize(payload), indent=2))


if __name__ == "__main__":
    main()

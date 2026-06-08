import json
import os
from collections import defaultdict
from datetime import datetime

STEPS = ["signup", "onboarding_complete", "first_action", "activated", "subscribed"]


def load_users(path):
    users = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            u = users.get(e["user_id"])
            if u is None:
                u = {"steps": {}, "platform": e.get("platform"), "country": e.get("country")}
                users[e["user_id"]] = u
            if e["event"] in STEPS:
                ts = datetime.fromisoformat(e["ts"])
                cur = u["steps"].get(e["event"])
                if cur is None or ts < cur:
                    u["steps"][e["event"]] = ts
    for u in users.values():
        sign = u["steps"].get("signup")
        u["cohort"] = cohort_label(sign) if sign else None
    return users


def cohort_label(ts):
    iso = ts.isocalendar()
    return "{}-W{:02d}".format(iso[0], iso[1])


def reached_depth(user):
    steps = user["steps"]
    if "signup" not in steps:
        return 0
    last_ts = steps["signup"]
    depth = 1
    for step in STEPS[1:]:
        ts = steps.get(step)
        if ts is None or ts < last_ts:
            break
        last_ts = ts
        depth += 1
    return depth


def funnel_counts(users):
    counts = [0] * len(STEPS)
    for u in users:
        d = reached_depth(u)
        for i in range(d):
            counts[i] += 1
    return counts


def funnel_table(counts):
    rows = []
    top = counts[0] or 1
    for i, step in enumerate(STEPS):
        prev = counts[i - 1] if i > 0 else None
        conv_prev = round(100.0 * counts[i] / prev, 1) if prev else None
        dropoff = (prev - counts[i]) if prev else 0
        drop_rate = round(100.0 * dropoff / prev, 1) if prev else None
        rows.append(
            {
                "step": step,
                "users": counts[i],
                "conversion_from_prev_pct": conv_prev,
                "dropoff_from_prev": dropoff,
                "drop_rate_from_prev_pct": drop_rate,
                "conversion_from_top_pct": round(100.0 * counts[i] / top, 1),
            }
        )
    return rows


def group_by(users, key):
    groups = defaultdict(list)
    for u in users.values():
        groups[u[key]].append(u)
    return groups


def build_metrics(path):
    users = load_users(path)
    all_users = list(users.values())
    overall = funnel_table(funnel_counts(all_users))

    cohorts = {}
    for label, group in sorted(group_by(users, "cohort").items()):
        cohorts[label] = funnel_table(funnel_counts(group))

    platforms = {}
    for label, group in sorted(group_by(users, "platform").items()):
        platforms[label] = funnel_table(funnel_counts(group))

    return {
        "funnel_steps": STEPS,
        "total_users": len(all_users),
        "overall": overall,
        "by_cohort": cohorts,
        "by_platform": platforms,
    }


def detect_anomalies(metrics, rel_threshold=0.8):
    overall = {r["step"]: r["conversion_from_prev_pct"] for r in metrics["overall"]}
    out = []
    for scope_key, scope_name in (("by_cohort", "cohort"), ("by_platform", "platform")):
        for label, rows in metrics[scope_key].items():
            for r in rows:
                seg = r["conversion_from_prev_pct"]
                ov = overall.get(r["step"])
                if seg is None or not ov:
                    continue
                ratio = seg / ov
                if ratio < rel_threshold:
                    out.append(
                        {
                            "scope": scope_name,
                            "label": label,
                            "step": r["step"],
                            "segment_conv": seg,
                            "overall_conv": ov,
                            "rel_drop_pct": round(100 * (1 - ratio), 1),
                        }
                    )
    out.sort(key=lambda a: a["segment_conv"] / a["overall_conv"])
    return out


def _bar(pct):
    return "#" * int(round(pct / 4.0))


def render_report(metrics):
    lines = []
    lines.append("FUNNEL  ({} users)".format(metrics["total_users"]))
    lines.append("-" * 60)
    for r in metrics["overall"]:
        cp = "" if r["conversion_from_prev_pct"] is None else " conv {:>5}%".format(
            r["conversion_from_prev_pct"]
        )
        lines.append(
            "{:<20} {:>5}  {:>5}% {:<25}{}".format(
                r["step"], r["users"], r["conversion_from_top_pct"],
                _bar(r["conversion_from_top_pct"]), cp,
            )
        )
    lines.append("")
    lines.append("DROP-OFF (step over step)")
    lines.append("-" * 60)
    for r in metrics["overall"][1:]:
        lines.append(
            "{:<20} -{:<5} users  ({}% drop)".format(
                r["step"], r["dropoff_from_prev"], r["drop_rate_from_prev_pct"]
            )
        )

    for title, key in (("BY COHORT (signup week)", "by_cohort"), ("BY PLATFORM", "by_platform")):
        lines.append("")
        lines.append(title)
        lines.append("-" * 60)
        header = "{:<14}".format("") + "".join("{:>11}".format(s[:10]) for s in STEPS)
        lines.append(header)
        for label, rows in metrics[key].items():
            cells = "".join("{:>11}".format(r["conversion_from_top_pct"]) for r in rows)
            lines.append("{:<14}{}".format(label, cells))
    return "\n".join(lines)


def main():
    path = os.path.join(os.path.dirname(__file__), "events.jsonl")
    metrics = build_metrics(path)
    print(render_report(metrics))


if __name__ == "__main__":
    main()

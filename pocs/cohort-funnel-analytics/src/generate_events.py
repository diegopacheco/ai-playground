import json
import os
import random
from datetime import datetime, timedelta

STEPS = ["signup", "onboarding_complete", "first_action", "activated", "subscribed"]
PLATFORMS = ["ios", "android", "web"]
COUNTRIES = ["US", "BR", "DE", "IN", "JP"]

BASE_PASS = {
    "onboarding_complete": 0.78,
    "first_action": 0.72,
    "activated": 0.63,
    "subscribed": 0.41,
}

COHORT_MOD = {
    0: {},
    1: {},
    2: {"onboarding_complete": 0.45},
    3: {"first_action": 0.85, "subscribed": 1.18},
}

PLATFORM_MOD = {
    "ios": {"subscribed": 1.2},
    "android": {"first_action": 0.62},
    "web": {},
}

USERS_PER_COHORT = 320
FIRST_MONDAY = datetime(2026, 1, 5, 9, 0, 0)


def pass_prob(step, cohort, platform):
    p = BASE_PASS[step]
    p *= COHORT_MOD.get(cohort, {}).get(step, 1.0)
    p *= PLATFORM_MOD.get(platform, {}).get(step, 1.0)
    return max(0.0, min(1.0, p))


def build_events(rng):
    events = []
    uid = 0
    for cohort in range(4):
        week_start = FIRST_MONDAY + timedelta(weeks=cohort)
        for _ in range(USERS_PER_COHORT):
            uid += 1
            user_id = "u{:05d}".format(uid)
            platform = rng.choices(PLATFORMS, weights=[0.4, 0.4, 0.2])[0]
            country = rng.choice(COUNTRIES)
            ts = week_start + timedelta(
                days=rng.randint(0, 6), hours=rng.randint(0, 12), minutes=rng.randint(0, 59)
            )
            ctx = {"platform": platform, "country": country}
            events.append(make_event(user_id, "signup", ts, ctx))
            if rng.random() < 0.5:
                events.append(
                    make_event(user_id, "app_open", ts + timedelta(minutes=rng.randint(1, 30)), ctx)
                )
            for step in STEPS[1:]:
                if rng.random() >= pass_prob(step, cohort, platform):
                    break
                ts = ts + timedelta(hours=rng.randint(1, 60), minutes=rng.randint(0, 59))
                events.append(make_event(user_id, step, ts, ctx))
    rng.shuffle(events)
    return events


def make_event(user_id, name, ts, ctx):
    e = {"user_id": user_id, "event": name, "ts": ts.isoformat()}
    e.update(ctx)
    return e


def main():
    rng = random.Random(42)
    events = build_events(rng)
    out = os.path.join(os.path.dirname(__file__), "events.jsonl")
    with open(out, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    print("wrote {} events for {} users to {}".format(len(events), 4 * USERS_PER_COHORT, out))


if __name__ == "__main__":
    main()

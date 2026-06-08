import os

import agent
import funnel
import generate_events

EVENTS = os.path.join(os.path.dirname(__file__), "events.jsonl")


def main():
    if not os.path.exists(EVENTS):
        generate_events.main()

    metrics = funnel.build_metrics(EVENTS)
    print(funnel.render_report(metrics))
    print()
    print("=" * 60)
    print("AGENT EXPLANATION")
    print("=" * 60)

    if not agent.has_key():
        print("OPENAI_API_KEY not set. Computed the funnel above but skipped the")
        print("agent reasoning step. Export the key and re-run to get the explanation.")
        return

    print(agent.explain(metrics))


if __name__ == "__main__":
    main()

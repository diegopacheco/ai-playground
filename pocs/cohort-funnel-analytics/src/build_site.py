import json
import os
import shutil
from datetime import datetime

import agent
import funnel
import generate_events

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
EVENTS = os.path.join(HERE, "events.jsonl")
TEMPLATE = os.path.join(HERE, "site_template.html")
SITE = os.path.join(ROOT, "site")


def count_lines(path):
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def main():
    if not os.path.exists(EVENTS):
        generate_events.main()

    metrics = funnel.build_metrics(EVENTS)
    anomalies = funnel.detect_anomalies(metrics)

    agent_text = None
    if agent.has_key():
        print("calling agent ({}) ...".format(agent.MODEL))
        agent_text = agent.explain(metrics)
    else:
        print("OPENAI_API_KEY not set; building site with the agent panel as a notice.")

    payload = {
        "metrics": metrics,
        "anomalies": anomalies,
        "agent_text": agent_text,
        "model": agent.MODEL,
        "event_count": count_lines(EVENTS),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    with open(TEMPLATE) as f:
        template = f.read()
    data_json = json.dumps(payload).replace("</", "<\\/")
    html = template.replace("__DATA_JSON__", data_json)

    os.makedirs(SITE, exist_ok=True)
    with open(os.path.join(SITE, "index.html"), "w") as f:
        f.write(html)
    shutil.copy(os.path.join(ROOT, "architecture.svg"), os.path.join(SITE, "architecture.svg"))
    print("wrote {}".format(os.path.join(SITE, "index.html")))


if __name__ == "__main__":
    main()

import json
import os

MODEL = "gpt-4o"

SYSTEM = (
    "You are a product analytics agent. You receive precomputed funnel, cohort, and "
    "segment metrics derived from a raw user-event log. The numbers are authoritative: "
    "never recompute or invent figures, reason only from what is given. Your job is to "
    "find where users fall out of the funnel and explain why, grounded strictly in the "
    "data. Compare cohorts and platforms against the overall funnel to isolate which "
    "segment drives each drop. Be specific and concise.\n\n"
    "Answer in this structure:\n"
    "1. Biggest drop-offs: the steps with the largest losses, with the numbers.\n"
    "2. Who falls out: the cohort(s) and platform(s) underperforming the overall rate.\n"
    "3. Likely why: a hypothesis for each drop, tied to the segment evidence.\n"
    "4. What to investigate next: concrete checks a team should run."
)


def explain(metrics):
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=4000,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": "Analyze this funnel.\n\n" + json.dumps(metrics, indent=2),
            },
        ],
    )
    return resp.choices[0].message.content


def has_key():
    return bool(os.environ.get("OPENAI_API_KEY"))

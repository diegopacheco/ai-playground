import json

from ollama import chat

from requests_data import AGENT_HINTS, AGENTS, SENTIMENTS

AGENT_SCHEMA = {
    "type": "object",
    "properties": {"agent": {"type": "string", "enum": AGENTS}},
    "required": ["agent"],
}
SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": SENTIMENTS}},
    "required": ["sentiment"],
}

AGENT_SYSTEM = ("You route customer messages to one agent.\n"
                + "\n".join(f"{agent}: {hint}" for agent, hint in AGENT_HINTS.items()))
SENTIMENT_SYSTEM = "You label the sentiment of a customer message."

AGENT_QUESTION = 'Message: "{text}"\nAgent?'
SENTIMENT_QUESTION = 'Message: "{text}"\nSentiment?'

AGENT_SHOTS = [
    ("My last invoice is wrong.", "billing"),
    ("The app crashes when I open settings.", "tech_support"),
    ("How much is the team plan?", "sales"),
    ("WIN a free laptop, click this link now!!!", "abuse"),
    ("Hi there, quick question.", "general"),
]
SENTIMENT_SHOTS = [
    ("This has been broken for three days.", "negative"),
    ("Where do I find the settings page?", "neutral"),
    ("The new release is excellent, thank you.", "positive"),
]


def _ask(text, model, system, question, shots, key, schema, fallback):
    messages = [{"role": "system", "content": system}]
    for shot, answer in shots:
        messages.append({"role": "user", "content": question.format(text=shot)})
        messages.append({"role": "assistant", "content": json.dumps({key: answer})})
    messages.append({"role": "user", "content": question.format(text=text)})
    response = chat(model=model, messages=messages, format=schema,
                    options={"temperature": 0, "num_predict": 16})
    try:
        return json.loads(response.message.content).get(key, fallback)
    except json.JSONDecodeError:
        return fallback


def route(text, model):
    agent = _ask(text, model, AGENT_SYSTEM, AGENT_QUESTION, AGENT_SHOTS,
                 "agent", AGENT_SCHEMA, "general")
    sentiment = _ask(text, model, SENTIMENT_SYSTEM, SENTIMENT_QUESTION, SENTIMENT_SHOTS,
                     "sentiment", SENTIMENT_SCHEMA, "neutral")
    return agent, sentiment

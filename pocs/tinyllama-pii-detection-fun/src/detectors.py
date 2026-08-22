import json
import re

from ollama import chat

from samples import LABELS

PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "PHONE": re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?){2,4}\d{2,4}"),
}
PHONE_LAST = ("EMAIL", "SSN", "CREDIT_CARD", "IP_ADDRESS")

ALL_SCHEMA = {
    "type": "object",
    "properties": {label: {"type": "boolean"} for label in LABELS},
    "required": LABELS,
}
BOOL_SCHEMA = {
    "type": "object",
    "properties": {"present": {"type": "boolean"}},
    "required": ["present"],
}

ALL_PROMPT = (
    "Does this text contain each kind of personal data? "
    "Answer true or false for each.\n\nText: {text}"
)

NAME_SYSTEM = (
    "You decide if a text mentions a person's full name (a first name followed "
    "by a last name). Products, cities, companies, emails and numbers are not "
    "person names."
)
NAME_QUESTION = 'Text: "{text}"\nDoes it mention a person\'s full name?'
NAME_SHOTS = [
    ("Deploy failed on the staging cluster, rolling back now.", False),
    ("Marcelo Gomes signed off on the migration plan.", True),
    ("The retry budget is 3 attempts with 200ms backoff.", False),
    ("Ana Lima approved the change.", True),
]


def detect_regex(text):
    found = {label for label in PHONE_LAST if PATTERNS[label].search(text)}
    stripped = text
    for label in PHONE_LAST:
        stripped = PATTERNS[label].sub(" ", stripped)
    if PATTERNS["PHONE"].search(stripped):
        found.add("PHONE")
    return found


def detect_llm(text, model):
    response = chat(
        model=model,
        messages=[{"role": "user", "content": ALL_PROMPT.format(text=text)}],
        format=ALL_SCHEMA,
        options={"temperature": 0},
    )
    payload = _load(response)
    return {label for label in LABELS if payload.get(label)}


def detect_name(text, model):
    messages = [{"role": "system", "content": NAME_SYSTEM}]
    for shot, present in NAME_SHOTS:
        messages.append({"role": "user", "content": NAME_QUESTION.format(text=shot)})
        messages.append({"role": "assistant", "content": json.dumps({"present": present})})
    messages.append({"role": "user", "content": NAME_QUESTION.format(text=text)})
    response = chat(model=model, messages=messages, format=BOOL_SCHEMA,
                    options={"temperature": 0})
    return bool(_load(response).get("present"))


def detect_hybrid(text, model):
    found = detect_regex(text)
    if detect_name(text, model):
        found.add("PERSON_NAME")
    return found


def _load(response):
    try:
        return json.loads(response.message.content)
    except json.JSONDecodeError:
        return {}

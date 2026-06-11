import os
import sys

import openai
from burr.core import ApplicationBuilder, State, action

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PROJECT = "apache-burr-poc"

_client = None


def client():
    global _client
    if _client is None:
        _client = openai.OpenAI()
    return _client


@action(reads=[], writes=["prompt", "chat_history"])
def human_input(state: State, prompt: str) -> State:
    return state.update(prompt=prompt).append(
        chat_history={"role": "user", "content": prompt}
    )


@action(reads=["chat_history"], writes=["response", "chat_history"])
def ai_response(state: State) -> State:
    content = (
        client()
        .chat.completions.create(model=MODEL, messages=state["chat_history"])
        .choices[0]
        .message.content
    )
    return state.update(response=content).append(
        chat_history={"role": "assistant", "content": content}
    )


def build_app():
    return (
        ApplicationBuilder()
        .with_actions(human_input, ai_response)
        .with_transitions(
            ("human_input", "ai_response"),
            ("ai_response", "human_input"),
        )
        .with_state(chat_history=[])
        .with_entrypoint("human_input")
        .with_tracker(project=PROJECT)
        .build()
    )


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-...", file=sys.stderr)
        return 1
    app = build_app()
    print(f"Apache Burr chatbot ready (model={MODEL}). Type 'exit' to quit.")
    print("Watch each run live in the Burr UI at http://localhost:7241\n")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            break
        *_, state = app.run(halt_after=["ai_response"], inputs={"prompt": prompt})
        print(f"bot> {state['response']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

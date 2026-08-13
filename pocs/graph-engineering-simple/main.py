import argparse

from agent import Agent
from graph import build_graph


def main():
    parser = argparse.ArgumentParser(description="Run a small graph of LLM calls")
    parser.add_argument("question")
    parser.add_argument("--provider", choices=["claude", "codex", "agy", "ollama"], default="ollama")
    parser.add_argument("--model", default="llama3.2")
    args = parser.parse_args()
    agent = Agent(args.provider)
    results = build_graph(lambda prompt: agent.call(args.model, prompt)).run(args.question)
    print(results["answer"])


if __name__ == "__main__":
    main()

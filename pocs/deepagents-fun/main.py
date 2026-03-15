from deepagents import create_deep_agent

def main():
    agent = create_deep_agent(model="openai:gpt-4o")
    result = agent.invoke({"messages": [{"role": "user", "content": "Research LangGraph and write a summary"}]})
    print(result)

if __name__ == "__main__":
    main()

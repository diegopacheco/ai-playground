import asyncio
from dapr_agents import Agent
from dapr_agents.llm import DaprChatClient
from tools import get_weather, get_time

async def main():
    llm = DaprChatClient(component_name="llm-provider")

    agent = Agent(
        name="AssistantAgent",
        role="General Assistant",
        instructions=[
            "You are a helpful assistant that can check weather and time.",
            "Be concise and friendly in your responses.",
        ],
        tools=[get_weather, get_time],
        llm=llm,
    )

    queries = [
        "What is the weather in New York?",
        "What time is it in PST?",
        "How about the weather in Tokyo and the time in JST?",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = await agent.run(query)
        print(f"Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())

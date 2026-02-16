import asyncio
from openmemory.client import Memory

async def main():
    mem = Memory()
    result = await mem.add("user prefers dark mode", user_id="u1")
    print("Added memory:", result)

    results = await mem.search("preferences", user_id="u1")
    print("Search results:", results)

    if result.get("id"):
        await mem.delete(result["id"])
        print("Deleted memory:", result["id"])

asyncio.run(main())

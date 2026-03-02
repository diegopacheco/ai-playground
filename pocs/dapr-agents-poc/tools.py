import random
from dapr_agents import tool

@tool
def get_weather(location: str) -> str:
    """Get weather information for a specific location."""
    conditions = ["sunny", "cloudy", "rainy", "windy", "snowy"]
    temperature = random.randint(50, 95)
    condition = random.choice(conditions)
    return f"{location}: {temperature}F, {condition}."

@tool
def get_time(timezone: str) -> str:
    """Get the current time for a given timezone."""
    from datetime import datetime, timezone as tz, timedelta
    offsets = {
        "UTC": 0, "EST": -5, "CST": -6, "MST": -7, "PST": -8,
        "CET": 1, "EET": 2, "IST": 5, "JST": 9, "AEST": 10,
        "BRT": -3, "GMT": 0,
    }
    offset = offsets.get(timezone.upper(), 0)
    now = datetime.now(tz(timedelta(hours=offset)))
    return f"Current time in {timezone}: {now.strftime('%Y-%m-%d %H:%M:%S')}"

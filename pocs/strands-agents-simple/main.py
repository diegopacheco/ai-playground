import os
import json
import urllib.request
import urllib.parse
from strands import Agent, tool
from strands.models.openai import OpenAIModel

CITY_COORDS = {
    "london": (51.5074, -0.1278),
    "tokyo": (35.6762, 139.6503),
    "new york": (40.7128, -74.0060),
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
    "sydney": (-33.8688, 151.2093),
    "sao paulo": (-23.5505, -46.6333),
    "mumbai": (19.0760, 72.8777),
    "beijing": (39.9042, 116.4074),
    "cairo": (30.0444, 31.2357),
}

def _geocode(city: str):
    coords = CITY_COORDS.get(city.lower())
    if coords:
        return coords
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1"
    with urllib.request.urlopen(geo_url, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    if "results" not in data or len(data["results"]) == 0:
        return None
    return (data["results"][0]["latitude"], data["results"][0]["longitude"])

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        str: A string with the current weather information.
    """
    try:
        coords = _geocode(city)
        if not coords:
            return f"Could not find coordinates for {city}"
        lat, lon = coords
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m,weather_code"
        )
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
        current = data["current"]
        temp = current["temperature_2m"]
        feels_like = current["apparent_temperature"]
        humidity = current["relative_humidity_2m"]
        wind_speed = current["wind_speed_10m"]
        wmo_code = current["weather_code"]
        description = _wmo_description(wmo_code)
        return (
            f"Weather in {city}: {description}, "
            f"Temperature: {temp}°C (feels like {feels_like}°C), "
            f"Humidity: {humidity}%, Wind: {wind_speed} km/h"
        )
    except Exception as e:
        return f"Could not retrieve weather for {city}: {str(e)}"

def _wmo_description(code: int) -> str:
    descriptions = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
    }
    return descriptions.get(code, f"Weather code {code}")

model = OpenAIModel(
    client_args={
        "api_key": os.environ.get("OPENAI_API_KEY"),
    },
    model_id="gpt-4o",
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    },
)

agent = Agent(model=model, tools=[get_weather])
response = agent("What is the weather in London and Tokyo?")
print(response)

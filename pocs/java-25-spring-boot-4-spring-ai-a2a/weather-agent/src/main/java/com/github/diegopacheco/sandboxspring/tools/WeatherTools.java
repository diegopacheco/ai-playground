package com.github.diegopacheco.sandboxspring.tools;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Service;

@Service
public class WeatherTools {

    @Tool(description = "Get current weather conditions for a location")
    public String getCurrentWeather(
            @ToolParam(description = "City and country, e.g. Paris, France") String location) {
        return "Current weather in " + location + ": Sunny, 22°C, humidity 60%, SW breeze at 10 km/h, UV index 5";
    }

    @Tool(description = "Get weather forecast for the next N days")
    public String getWeatherForecast(
            @ToolParam(description = "City and country, e.g. Paris, France") String location,
            @ToolParam(description = "Number of days for forecast, 1 to 7") int days) {
        return "Weather forecast for " + location + " (" + days + " days): "
                + "Day 1: Sunny 24°C, Day 2: Partly cloudy 21°C, Day 3: Rainy 17°C, "
                + "Day 4-7: Mixed sun and clouds 18-23°C. Pack a light jacket for evenings.";
    }
}

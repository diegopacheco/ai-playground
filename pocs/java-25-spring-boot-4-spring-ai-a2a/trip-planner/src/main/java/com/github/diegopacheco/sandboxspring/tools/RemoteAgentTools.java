package com.github.diegopacheco.sandboxspring.tools;

import com.github.diegopacheco.sandboxspring.client.A2AClient;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class RemoteAgentTools {

    private final A2AClient a2aClient;

    @Value("${agents.weather.url}")
    private String weatherAgentUrl;

    @Value("${agents.hotel.url}")
    private String hotelAgentUrl;

    public RemoteAgentTools(A2AClient a2aClient) {
        this.a2aClient = a2aClient;
    }

    @Tool(description = "Get weather information and forecast for a travel destination by calling the Weather Agent via A2A protocol")
    public String getWeatherForDestination(
            @ToolParam(description = "Travel destination city and country, e.g. Paris, France") String destination) {
        return a2aClient.sendMessage(weatherAgentUrl,
                "Provide current weather and 7-day forecast for " + destination);
    }

    @Tool(description = "Search for hotels and accommodations at a travel destination by calling the Hotel Agent via A2A protocol")
    public String searchHotelsAtDestination(
            @ToolParam(description = "Travel destination city and country, e.g. Rome, Italy") String destination,
            @ToolParam(description = "Budget preference: budget, mid-range, or luxury") String budget) {
        return a2aClient.sendMessage(hotelAgentUrl,
                "Find " + budget + " hotels in " + destination + " with availability and pricing details");
    }
}

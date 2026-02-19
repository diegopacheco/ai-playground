package com.github.diegopacheco.sandboxspring.tools;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Service;

@Service
public class HotelTools {

    @Tool(description = "Search for hotels in a destination by budget tier")
    public String searchHotels(
            @ToolParam(description = "City and country, e.g. Rome, Italy") String destination,
            @ToolParam(description = "Budget tier: budget, mid-range, or luxury") String budgetTier) {
        return switch (budgetTier.toLowerCase()) {
            case "budget" -> "Budget hotels in " + destination + ": City Hostel (€35/night, 3km from center), Budget Inn (€55/night, central location), Travelers Rest (€45/night, free breakfast)";
            case "luxury" -> "Luxury hotels in " + destination + ": Grand Palace Hotel (€350/night, 5-star, rooftop pool), The Royal Suite (€500/night, butler service), Prestige Collection (€420/night, spa included)";
            default -> "Mid-range hotels in " + destination + ": Hotel Comfort Plus (€120/night, 4-star), Business Traveler Inn (€90/night, free breakfast), City Center Suites (€110/night, kitchen)";
        };
    }

    @Tool(description = "Check hotel availability and pricing for specific dates")
    public String checkAvailability(
            @ToolParam(description = "Hotel name") String hotelName,
            @ToolParam(description = "Check-in date in YYYY-MM-DD format") String checkIn,
            @ToolParam(description = "Check-out date in YYYY-MM-DD format") String checkOut) {
        return hotelName + ": Available " + checkIn + " to " + checkOut
                + ". Standard room €150/night, Superior €200/night, Suite €280/night. "
                + "Free cancellation up to 48h before check-in. Breakfast included.";
    }

    @Tool(description = "Get hotel amenities and features for a specific hotel")
    public String getHotelAmenities(
            @ToolParam(description = "Hotel name") String hotelName) {
        return hotelName + " amenities: Free WiFi, Restaurant & Bar, Spa & Fitness Center, "
                + "Business Center, 24h Concierge, Parking (€20/day), Pet-friendly rooms, Airport shuttle";
    }
}

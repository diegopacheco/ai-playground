package com.github.diegopacheco.sandboxspring.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/plan")
public class TripPlannerController {

    private final ChatClient tripPlannerChatClient;

    public TripPlannerController(ChatClient tripPlannerChatClient) {
        this.tripPlannerChatClient = tripPlannerChatClient;
    }

    @PostMapping
    public String planTrip(@RequestBody TripRequest request) {
        return tripPlannerChatClient.prompt()
                .user("Plan a " + request.days() + " day trip to " + request.destination()
                        + " with a " + request.budget() + " budget")
                .call()
                .content();
    }

    record TripRequest(String destination, int days, String budget) {}
}

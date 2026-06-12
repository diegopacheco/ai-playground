package com.bookstore.service;

import java.util.UUID;
import org.springframework.stereotype.Service;

@Service
public class GatewayClient {

    public String submit(String customerId, double amount) {
        return sign(customerId + ":" + amount);
    }

    private String sign(String payload) {
        return UUID.nameUUIDFromBytes(payload.getBytes()).toString();
    }
}

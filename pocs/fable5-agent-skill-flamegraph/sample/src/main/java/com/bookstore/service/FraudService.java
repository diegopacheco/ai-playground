package com.bookstore.service;

import org.springframework.stereotype.Service;

@Service
public class FraudService {

    public void assess(String customerId, double amount) {
        double score = scoreAmount(amount) + scoreVelocity(customerId);
        if (score > 0.9) {
            throw new IllegalStateException("payment flagged for " + customerId);
        }
    }

    private double scoreAmount(double amount) {
        return amount > 500 ? 0.6 : 0.1;
    }

    private double scoreVelocity(String customerId) {
        return customerId.length() > 24 ? 0.5 : 0.1;
    }
}

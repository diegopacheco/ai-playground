package com.bookstore.service;

import org.springframework.stereotype.Service;

@Service
public class DiscountService {

    public double discountFor(String customerId, double subtotal) {
        double rate = loyaltyRate(customerId) + seasonalRate();
        return subtotal * Math.min(rate, 0.25);
    }

    private double loyaltyRate(String customerId) {
        return customerId.hashCode() % 2 == 0 ? 0.10 : 0.05;
    }

    private double seasonalRate() {
        return 0.05;
    }
}

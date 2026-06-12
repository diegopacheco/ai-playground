package com.bookstore.service;

import org.springframework.stereotype.Service;

@Service
public class TaxService {

    public double taxFor(double amount, String region) {
        return amount * rateForRegion(region);
    }

    private double rateForRegion(String region) {
        return switch (region) {
            case "CA" -> 0.0925;
            case "NY" -> 0.08875;
            default -> 0.07;
        };
    }
}

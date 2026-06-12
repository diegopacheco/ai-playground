package com.bookstore.service;

import org.springframework.stereotype.Service;

@Service
public class CarrierService {

    public String book(String customerId) {
        return labelFor(customerId);
    }

    private String labelFor(String customerId) {
        return "TRK-" + Math.abs(customerId.hashCode());
    }
}

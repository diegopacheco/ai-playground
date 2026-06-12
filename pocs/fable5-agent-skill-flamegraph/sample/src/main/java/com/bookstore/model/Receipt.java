package com.bookstore.model;

import java.time.Instant;

public record Receipt(String orderId, String customerId, double total, String trackingCode, Instant issuedAt) {
}

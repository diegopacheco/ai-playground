package com.store.model;

public record OrderLine(String productId, int quantity, long unitPriceCents) {
}

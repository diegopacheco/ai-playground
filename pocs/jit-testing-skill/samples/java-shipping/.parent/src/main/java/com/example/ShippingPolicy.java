package com.example;

public final class ShippingPolicy {
    private static final int FREE_SHIPPING_THRESHOLD_CENTS = 5000;

    public boolean qualifiesForFreeShipping(Order order) {
        return order.totalCents() >= FREE_SHIPPING_THRESHOLD_CENTS;
    }
}

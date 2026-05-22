package com.example;

public class Order {
    private final int totalCents;
    private final boolean pickup;

    public Order(int totalCents, boolean pickup) {
        this.totalCents = totalCents;
        this.pickup = pickup;
    }

    public int totalCents() { return totalCents; }
    public boolean pickup() { return pickup; }
}

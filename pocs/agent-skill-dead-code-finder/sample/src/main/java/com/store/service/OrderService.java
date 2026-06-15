package com.store.service;

import com.store.model.Order;
import com.store.model.OrderLine;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.UUID;

@Service
public class OrderService {

    private final PricingService pricing;
    private final InventoryService inventory;
    private final NotificationService notifications;

    public OrderService(PricingService pricing, InventoryService inventory, NotificationService notifications) {
        this.pricing = pricing;
        this.inventory = inventory;
        this.notifications = notifications;
    }

    public Order place(List<OrderLine> lines) {
        if (!inventory.reserve(lines)) {
            throw new IllegalStateException("out of stock");
        }
        long total = pricing.total(lines);
        String id = UUID.randomUUID().toString();
        notifications.confirm(id);
        return new Order(id, lines, total);
    }

    public Order find(String id) {
        return new Order(id, List.of(), 0);
    }

    public void cancelAll() {
        System.out.println("cancel every open order");
    }
}

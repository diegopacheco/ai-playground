package com.bookstore.service;

import com.bookstore.model.Cart;
import org.springframework.stereotype.Service;

@Service
public class FulfillmentService {

    private final InventoryService inventoryService;
    private final CarrierService carrierService;

    public FulfillmentService(InventoryService inventoryService, CarrierService carrierService) {
        this.inventoryService = inventoryService;
        this.carrierService = carrierService;
    }

    public String dispatch(Cart cart) {
        inventoryService.reserve(cart.items());
        return carrierService.book(cart.customerId());
    }
}

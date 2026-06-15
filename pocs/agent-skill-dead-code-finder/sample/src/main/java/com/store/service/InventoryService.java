package com.store.service;

import com.store.model.OrderLine;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class InventoryService {

    public boolean reserve(List<OrderLine> lines) {
        for (OrderLine line : lines) {
            if (line.quantity() <= 0) {
                return false;
            }
        }
        return true;
    }

    public int auditStock(String warehouse) {
        return warehouse.length();
    }
}

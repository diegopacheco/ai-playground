package com.bookstore.service;

import com.bookstore.model.Receipt;
import org.springframework.stereotype.Service;

@Service
public class NotificationService {

    public void email(Receipt receipt) {
        String message = render(receipt);
        System.out.println("sending to " + receipt.customerId() + ": " + message);
    }

    private String render(Receipt receipt) {
        return "order %s total %.2f tracking %s".formatted(
                receipt.orderId(), receipt.total(), receipt.trackingCode());
    }
}

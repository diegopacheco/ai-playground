package com.bookstore.service;

import com.bookstore.model.PricedCart;
import com.bookstore.model.Receipt;
import java.time.Instant;
import java.util.UUID;
import org.springframework.stereotype.Service;

@Service
public class ReceiptService {

    private final NotificationService notificationService;

    public ReceiptService(NotificationService notificationService) {
        this.notificationService = notificationService;
    }

    public Receipt issue(PricedCart pricedCart, String trackingCode) {
        Receipt receipt = new Receipt(
                UUID.randomUUID().toString(),
                pricedCart.cart().customerId(),
                pricedCart.total(),
                trackingCode,
                Instant.now());
        notificationService.email(receipt);
        return receipt;
    }
}

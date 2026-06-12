package com.bookstore.service;

import com.bookstore.model.PricedCart;
import org.springframework.stereotype.Service;

@Service
public class PaymentService {

    private final FraudService fraudService;
    private final GatewayClient gatewayClient;

    public PaymentService(FraudService fraudService, GatewayClient gatewayClient) {
        this.fraudService = fraudService;
        this.gatewayClient = gatewayClient;
    }

    public String charge(PricedCart pricedCart) {
        fraudService.assess(pricedCart.cart().customerId(), pricedCart.total());
        return gatewayClient.submit(pricedCart.cart().customerId(), pricedCart.total());
    }
}

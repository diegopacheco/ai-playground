package com.bookstore.service;

import com.bookstore.model.Cart;
import com.bookstore.model.PricedCart;
import com.bookstore.model.Receipt;
import org.springframework.stereotype.Service;

@Service
public class CheckoutService {

    private final CartService cartService;
    private final PricingService pricingService;
    private final PaymentService paymentService;
    private final FulfillmentService fulfillmentService;
    private final ReceiptService receiptService;

    public CheckoutService(CartService cartService, PricingService pricingService,
                           PaymentService paymentService, FulfillmentService fulfillmentService,
                           ReceiptService receiptService) {
        this.cartService = cartService;
        this.pricingService = pricingService;
        this.paymentService = paymentService;
        this.fulfillmentService = fulfillmentService;
        this.receiptService = receiptService;
    }

    public Receipt checkout(String customerId, String region) {
        Cart cart = cartService.loadCart(customerId);
        PricedCart pricedCart = pricingService.priceCart(cart, region);
        paymentService.charge(pricedCart);
        String trackingCode = fulfillmentService.dispatch(cart);
        return receiptService.issue(pricedCart, trackingCode);
    }
}

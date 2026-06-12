package com.bookstore.service;

import com.bookstore.model.Cart;
import com.bookstore.model.CartItem;
import com.bookstore.model.PricedCart;
import org.springframework.stereotype.Service;

@Service
public class PricingService {

    private final CatalogService catalogService;
    private final DiscountService discountService;
    private final TaxService taxService;

    public PricingService(CatalogService catalogService, DiscountService discountService, TaxService taxService) {
        this.catalogService = catalogService;
        this.discountService = discountService;
        this.taxService = taxService;
    }

    public PricedCart priceCart(Cart cart, String region) {
        double subtotal = 0;
        for (CartItem item : cart.items()) {
            subtotal += priceItem(item);
        }
        double discount = discountService.discountFor(cart.customerId(), subtotal);
        double tax = taxService.taxFor(subtotal - discount, region);
        return new PricedCart(cart, subtotal, discount, tax, subtotal - discount + tax);
    }

    private double priceItem(CartItem item) {
        return catalogService.priceOf(item.bookId()) * item.quantity();
    }
}

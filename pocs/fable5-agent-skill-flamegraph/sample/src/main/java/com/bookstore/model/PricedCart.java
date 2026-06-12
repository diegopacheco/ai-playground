package com.bookstore.model;

public record PricedCart(Cart cart, double subtotal, double discount, double tax, double total) {
}

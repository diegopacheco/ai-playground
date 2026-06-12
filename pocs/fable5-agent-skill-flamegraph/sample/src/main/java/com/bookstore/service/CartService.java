package com.bookstore.service;

import com.bookstore.model.Cart;
import com.bookstore.model.CartItem;
import com.bookstore.repository.CartRepository;
import org.springframework.stereotype.Service;

@Service
public class CartService {

    private final CartRepository cartRepository;
    private final InventoryService inventoryService;

    public CartService(CartRepository cartRepository, InventoryService inventoryService) {
        this.cartRepository = cartRepository;
        this.inventoryService = inventoryService;
    }

    public Cart loadCart(String customerId) {
        Cart cart = cartRepository.findByCustomer(customerId);
        validateItems(cart);
        return cart;
    }

    private void validateItems(Cart cart) {
        for (CartItem item : cart.items()) {
            if (!inventoryService.checkStock(item.bookId(), item.quantity())) {
                throw new IllegalStateException("out of stock: " + item.bookId());
            }
        }
    }
}

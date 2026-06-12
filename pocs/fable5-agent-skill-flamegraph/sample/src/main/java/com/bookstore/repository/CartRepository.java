package com.bookstore.repository;

import com.bookstore.model.Cart;
import com.bookstore.model.CartItem;
import java.util.List;
import org.springframework.stereotype.Repository;

@Repository
public class CartRepository {

    public Cart findByCustomer(String customerId) {
        return new Cart(customerId, List.of(
                new CartItem("b1", 1),
                new CartItem("b2", 2),
                new CartItem("b4", 1)));
    }
}

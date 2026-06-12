package com.bookstore.service;

import com.bookstore.model.CartItem;
import com.bookstore.repository.StockRepository;
import java.util.List;
import org.springframework.stereotype.Service;

@Service
public class InventoryService {

    private final StockRepository stockRepository;

    public InventoryService(StockRepository stockRepository) {
        this.stockRepository = stockRepository;
    }

    public boolean checkStock(String bookId, int quantity) {
        return stockRepository.stockOf(bookId) >= quantity;
    }

    public void reserve(List<CartItem> items) {
        for (CartItem item : items) {
            stockRepository.deduct(item.bookId(), item.quantity());
        }
    }
}

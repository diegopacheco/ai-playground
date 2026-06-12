package com.bookstore.repository;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.stereotype.Repository;

@Repository
public class StockRepository {

    private final Map<String, Integer> stock = new ConcurrentHashMap<>(
            Map.of("b1", 12, "b2", 7, "b3", 3, "b4", 9));

    public int stockOf(String bookId) {
        return stock.getOrDefault(bookId, 0);
    }

    public void deduct(String bookId, int quantity) {
        stock.merge(bookId, -quantity, Integer::sum);
    }
}

package com.stock.repository;

import com.stock.model.Stock;
import org.springframework.stereotype.Repository;

import java.util.*;

@Repository
public class StockRepository {

    private final Map<String, Stock> stocks = new LinkedHashMap<>();

    public StockRepository() {
        stocks.put("AAPL", new Stock("AAPL", "Apple Inc.", 189.50, 2.30, 1.23, 54000000, 191.00, 187.20, 188.00, 187.20));
        stocks.put("GOOGL", new Stock("GOOGL", "Alphabet Inc.", 141.80, -1.20, -0.84, 28000000, 143.50, 140.10, 142.90, 143.00));
        stocks.put("MSFT", new Stock("MSFT", "Microsoft Corp.", 378.90, 4.50, 1.20, 32000000, 380.00, 375.00, 376.00, 374.40));
        stocks.put("AMZN", new Stock("AMZN", "Amazon.com Inc.", 178.25, 3.10, 1.77, 45000000, 179.80, 175.50, 176.00, 175.15));
        stocks.put("TSLA", new Stock("TSLA", "Tesla Inc.", 248.50, -5.30, -2.09, 98000000, 255.00, 246.00, 253.00, 253.80));
        stocks.put("NVDA", new Stock("NVDA", "NVIDIA Corp.", 875.30, 12.40, 1.44, 41000000, 880.00, 862.00, 865.00, 862.90));
        stocks.put("META", new Stock("META", "Meta Platforms Inc.", 505.75, 8.20, 1.65, 22000000, 508.00, 498.00, 500.00, 497.55));
        stocks.put("JPM", new Stock("JPM", "JPMorgan Chase", 198.40, 1.80, 0.92, 12000000, 199.50, 196.00, 197.00, 196.60));
    }

    public List<Stock> findAll() {
        return new ArrayList<>(stocks.values());
    }

    public Stock findBySymbol(String symbol) {
        return stocks.get(symbol.toUpperCase());
    }

    public void save(Stock stock) {
        stocks.put(stock.getSymbol(), stock);
    }

    public void delete(String symbol) {
        stocks.remove(symbol.toUpperCase());
    }
}

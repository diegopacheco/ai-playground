package com.stock.service;

import com.stock.model.Stock;
import com.stock.repository.StockRepository;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

@Service
public class StockService {

    private final StockRepository stockRepository;
    private final Random random = new Random();

    public StockService(StockRepository stockRepository) {
        this.stockRepository = stockRepository;
    }

    public List<Stock> getAllStocks() {
        return stockRepository.findAll();
    }

    public Stock getStockBySymbol(String symbol) {
        return stockRepository.findBySymbol(symbol);
    }

    public List<Stock> getGainers() {
        return stockRepository.findAll().stream()
                .filter(s -> s.getChange() > 0)
                .sorted((a, b) -> Double.compare(b.getChangePercent(), a.getChangePercent()))
                .collect(Collectors.toList());
    }

    public List<Stock> getLosers() {
        return stockRepository.findAll().stream()
                .filter(s -> s.getChange() < 0)
                .sorted(Comparator.comparingDouble(Stock::getChangePercent))
                .collect(Collectors.toList());
    }

    public List<Stock> getTopVolume() {
        return stockRepository.findAll().stream()
                .sorted((a, b) -> Long.compare(b.getVolume(), a.getVolume()))
                .collect(Collectors.toList());
    }

    public Map<String, Object> getDashboardStats() {
        List<Stock> stocks = stockRepository.findAll();
        Map<String, Object> stats = new HashMap<>();
        double totalValue = 0;
        double totalChange = 0;
        long totalVolume = 0;
        int gainerCount = 0;
        int loserCount = 0;

        for (Stock s : stocks) {
            totalValue += s.getPrice();
            totalChange += s.getChangePercent();
            totalVolume += s.getVolume();
            if (s.getChange() > 0) gainerCount++;
            else if (s.getChange() < 0) loserCount++;
        }

        stats.put("totalValue", String.format("%.2f", totalValue));
        stats.put("avgChange", stocks.isEmpty() ? 0.0 : totalChange / stocks.size());
        stats.put("totalVolume", totalVolume);
        stats.put("gainerCount", gainerCount);
        stats.put("loserCount", loserCount);
        stats.put("totalStocks", stocks.size());
        return stats;
    }

    public void simulatePriceUpdate() {
        for (Stock s : stockRepository.findAll()) {
            double changeAmount = (random.nextDouble() * 10) - 5;
            double newPrice = Math.max(1, s.getPrice() + changeAmount);
            Stock updated = new Stock(s);
            updated.setPrice(Math.round(newPrice * 100.0) / 100.0);
            updated.setChange(Math.round(changeAmount * 100.0) / 100.0);
            updated.setChangePercent(Math.round((changeAmount / s.getPrice()) * 10000.0) / 100.0);
            updated.setVolume(s.getVolume() + random.nextInt(1000000));
            updated.setHigh(Math.max(s.getHigh(), newPrice));
            updated.setLow(Math.min(s.getLow(), newPrice));
            stockRepository.save(updated);
        }
    }
}

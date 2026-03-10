package com.stock.controller;

import com.stock.model.Stock;
import com.stock.repository.StockRepository;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/api")
public class StockApiController {

    private final StockRepository stockRepository;

    public StockApiController(StockRepository stockRepository) {
        this.stockRepository = stockRepository;
    }

    @GetMapping("/stocks")
    public List<Map<String, Object>> getAllStocks() {
        List<Stock> stocks = stockRepository.findAll();
        List<Map<String, Object>> result = new ArrayList<>();
        for (int i = 0; i < stocks.size(); i++) {
            Stock s = stocks.get(i);
            Map<String, Object> map = new HashMap<>();
            map.put("symbol", s.getSymbol());
            map.put("name", s.getName());
            map.put("price", s.getPrice());
            map.put("change", s.getChange());
            map.put("changePercent", s.getChangePercent());
            map.put("volume", s.getVolume());
            map.put("high", s.getHigh());
            map.put("low", s.getLow());
            map.put("open", s.getOpen());
            map.put("previousClose", s.getPreviousClose());
            result.add(map);
        }
        return result;
    }

    @GetMapping("/stocks/{symbol}")
    public Map<String, Object> getStock(@PathVariable String symbol) {
        Stock s = stockRepository.findBySymbol(symbol);
        if (s == null) return Collections.emptyMap();
        Map<String, Object> map = new HashMap<>();
        map.put("symbol", s.getSymbol());
        map.put("name", s.getName());
        map.put("price", s.getPrice());
        map.put("change", s.getChange());
        map.put("changePercent", s.getChangePercent());
        map.put("volume", s.getVolume());
        map.put("high", s.getHigh());
        map.put("low", s.getLow());
        map.put("open", s.getOpen());
        map.put("previousClose", s.getPreviousClose());
        return map;
    }

    @GetMapping("/gainers")
    public List<Map<String, Object>> getGainers() {
        List<Stock> stocks = stockRepository.findAll();
        List<Map<String, Object>> result = new ArrayList<>();
        for (int i = 0; i < stocks.size(); i++) {
            Stock s = stocks.get(i);
            if (s.getChange() > 0) {
                Map<String, Object> map = new HashMap<>();
                map.put("symbol", s.getSymbol());
                map.put("name", s.getName());
                map.put("price", s.getPrice());
                map.put("change", s.getChange());
                map.put("changePercent", s.getChangePercent());
                map.put("volume", s.getVolume());
                map.put("high", s.getHigh());
                map.put("low", s.getLow());
                map.put("open", s.getOpen());
                map.put("previousClose", s.getPreviousClose());
                result.add(map);
            }
        }
        result.sort((a, b) -> Double.compare((double) b.get("changePercent"), (double) a.get("changePercent")));
        return result;
    }

    @GetMapping("/losers")
    public List<Map<String, Object>> getLosers() {
        List<Stock> stocks = stockRepository.findAll();
        List<Map<String, Object>> result = new ArrayList<>();
        for (int i = 0; i < stocks.size(); i++) {
            Stock s = stocks.get(i);
            if (s.getChange() < 0) {
                Map<String, Object> map = new HashMap<>();
                map.put("symbol", s.getSymbol());
                map.put("name", s.getName());
                map.put("price", s.getPrice());
                map.put("change", s.getChange());
                map.put("changePercent", s.getChangePercent());
                map.put("volume", s.getVolume());
                map.put("high", s.getHigh());
                map.put("low", s.getLow());
                map.put("open", s.getOpen());
                map.put("previousClose", s.getPreviousClose());
                result.add(map);
            }
        }
        result.sort((a, b) -> Double.compare((double) a.get("changePercent"), (double) b.get("changePercent")));
        return result;
    }

    @GetMapping("/stats")
    public Map<String, Object> getStats() {
        List<Stock> stocks = stockRepository.findAll();
        Map<String, Object> stats = new HashMap<>();

        double totalValue = 0;
        for (int i = 0; i < stocks.size(); i++) {
            totalValue = totalValue + stocks.get(i).getPrice();
        }
        stats.put("totalValue", totalValue);

        double avgChange = 0;
        for (int i = 0; i < stocks.size(); i++) {
            avgChange = avgChange + stocks.get(i).getChangePercent();
        }
        avgChange = avgChange / stocks.size();
        stats.put("avgChange", avgChange);

        long totalVol = 0;
        for (int i = 0; i < stocks.size(); i++) {
            totalVol = totalVol + stocks.get(i).getVolume();
        }
        stats.put("totalVolume", totalVol);

        int gainerCount = 0;
        for (int i = 0; i < stocks.size(); i++) {
            if (stocks.get(i).getChange() > 0) {
                gainerCount++;
            }
        }
        stats.put("gainerCount", gainerCount);

        int loserCount = 0;
        for (int i = 0; i < stocks.size(); i++) {
            if (stocks.get(i).getChange() < 0) {
                loserCount++;
            }
        }
        stats.put("loserCount", loserCount);

        stats.put("totalStocks", stocks.size());
        return stats;
    }

    @PostMapping("/simulate")
    public Map<String, String> simulate() {
        List<Stock> stocks = stockRepository.findAll();
        Random random = new Random();
        for (int i = 0; i < stocks.size(); i++) {
            Stock s = stocks.get(i);
            double changeAmount = (random.nextDouble() * 10) - 5;
            double newPrice = s.getPrice() + changeAmount;
            if (newPrice < 1) newPrice = 1;

            Stock updated = new Stock();
            updated.setSymbol(s.getSymbol());
            updated.setName(s.getName());
            updated.setPrice(Math.round(newPrice * 100.0) / 100.0);
            updated.setChange(Math.round(changeAmount * 100.0) / 100.0);
            updated.setChangePercent(Math.round((changeAmount / s.getPrice()) * 10000.0) / 100.0);
            updated.setVolume(s.getVolume() + random.nextInt(1000000));
            updated.setHigh(Math.max(s.getHigh(), newPrice));
            updated.setLow(Math.min(s.getLow(), newPrice));
            updated.setOpen(s.getOpen());
            updated.setPreviousClose(s.getPreviousClose());
            stockRepository.save(updated);
        }
        Map<String, String> response = new HashMap<>();
        response.put("status", "ok");
        response.put("message", "Prices updated");
        return response;
    }
}

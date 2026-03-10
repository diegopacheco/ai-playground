package com.stock.controller;

import com.stock.model.Stock;
import com.stock.service.StockService;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/api")
public class StockApiController {

    private final StockService stockService;

    public StockApiController(StockService stockService) {
        this.stockService = stockService;
    }

    @GetMapping("/stocks")
    public List<Stock> getAllStocks() {
        return stockService.getAllStocks();
    }

    @GetMapping("/stocks/{symbol}")
    public Stock getStock(@PathVariable String symbol) {
        return stockService.getStockBySymbol(symbol);
    }

    @GetMapping("/gainers")
    public List<Stock> getGainers() {
        return stockService.getGainers();
    }

    @GetMapping("/losers")
    public List<Stock> getLosers() {
        return stockService.getLosers();
    }

    @GetMapping("/stats")
    public Map<String, Object> getStats() {
        return stockService.getDashboardStats();
    }

    @PostMapping("/simulate")
    public Map<String, String> simulate() {
        stockService.simulatePriceUpdate();
        Map<String, String> response = new HashMap<>();
        response.put("status", "ok");
        response.put("message", "Prices updated");
        return response;
    }
}

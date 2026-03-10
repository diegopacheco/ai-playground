package com.stock.controller;

import com.stock.model.Stock;
import com.stock.service.StockService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;

import java.util.Map;

@Controller
public class DashboardController {

    private final StockService stockService;

    public DashboardController(StockService stockService) {
        this.stockService = stockService;
    }

    @GetMapping("/")
    public String dashboard(Model model) {
        model.addAttribute("stocks", stockService.getAllStocks());
        Map<String, Object> stats = stockService.getDashboardStats();
        stats.forEach(model::addAttribute);
        return "dashboard";
    }

    @GetMapping("/stock/{symbol}")
    public String stockDetail(@PathVariable String symbol, Model model) {
        Stock stock = stockService.getStockBySymbol(symbol);
        if (stock == null) {
            return "redirect:/";
        }
        model.addAttribute("stock", stock);
        double dayRange = stock.getHigh() - stock.getLow();
        model.addAttribute("dayRange", String.format("%.2f", dayRange));
        double changeFromOpen = stock.getPrice() - stock.getOpen();
        model.addAttribute("changeFromOpen", String.format("%.2f", changeFromOpen));
        double changeFromOpenPercent = (changeFromOpen / stock.getOpen()) * 100;
        model.addAttribute("changeFromOpenPercent", String.format("%.2f", changeFromOpenPercent));
        return "stock-detail";
    }

    @GetMapping("/gainers")
    public String gainers(Model model) {
        model.addAttribute("stocks", stockService.getGainers());
        model.addAttribute("title", "Top Gainers");
        return "stock-list";
    }

    @GetMapping("/losers")
    public String losers(Model model) {
        model.addAttribute("stocks", stockService.getLosers());
        model.addAttribute("title", "Top Losers");
        return "stock-list";
    }

    @GetMapping("/volume")
    public String topVolume(Model model) {
        model.addAttribute("stocks", stockService.getTopVolume());
        model.addAttribute("title", "Top Volume");
        return "stock-list";
    }
}

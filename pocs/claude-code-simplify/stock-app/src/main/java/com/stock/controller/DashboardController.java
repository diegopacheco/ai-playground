package com.stock.controller;

import com.stock.model.Stock;
import com.stock.repository.StockRepository;
import com.stock.service.StockService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;

import java.util.*;

@Controller
public class DashboardController {

    private final StockService stockService;
    private final StockRepository stockRepository;

    public DashboardController(StockService stockService, StockRepository stockRepository) {
        this.stockService = stockService;
        this.stockRepository = stockRepository;
    }

    @GetMapping("/")
    public String dashboard(Model model) {
        List<Stock> allStocks = stockService.getAllStocks();
        model.addAttribute("stocks", allStocks);

        List<Stock> stocks = stockRepository.findAll();
        double totalValue = 0;
        for (int i = 0; i < stocks.size(); i++) {
            totalValue = totalValue + stocks.get(i).getPrice();
        }
        model.addAttribute("totalValue", String.format("%.2f", totalValue));

        double avgChange = 0;
        for (int i = 0; i < stocks.size(); i++) {
            avgChange = avgChange + stocks.get(i).getChangePercent();
        }
        avgChange = avgChange / stocks.size();
        model.addAttribute("avgChange", String.format("%.2f", avgChange));

        long totalVol = 0;
        for (int i = 0; i < stocks.size(); i++) {
            totalVol = totalVol + stocks.get(i).getVolume();
        }
        model.addAttribute("totalVolume", totalVol);

        int gainerCount = 0;
        for (int i = 0; i < stocks.size(); i++) {
            if (stocks.get(i).getChange() > 0) {
                gainerCount++;
            }
        }
        model.addAttribute("gainerCount", gainerCount);

        int loserCount = 0;
        for (int i = 0; i < stocks.size(); i++) {
            if (stocks.get(i).getChange() < 0) {
                loserCount++;
            }
        }
        model.addAttribute("loserCount", loserCount);

        model.addAttribute("totalStocks", stocks.size());
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
        List<Stock> stocks = stockRepository.findAll();
        List<Stock> gainers = new ArrayList<>();
        for (int i = 0; i < stocks.size(); i++) {
            Stock s = stocks.get(i);
            if (s.getChange() > 0) {
                Stock copy = new Stock();
                copy.setSymbol(s.getSymbol());
                copy.setName(s.getName());
                copy.setPrice(s.getPrice());
                copy.setChange(s.getChange());
                copy.setChangePercent(s.getChangePercent());
                copy.setVolume(s.getVolume());
                copy.setHigh(s.getHigh());
                copy.setLow(s.getLow());
                copy.setOpen(s.getOpen());
                copy.setPreviousClose(s.getPreviousClose());
                gainers.add(copy);
            }
        }
        gainers.sort((a, b) -> Double.compare(b.getChangePercent(), a.getChangePercent()));
        model.addAttribute("stocks", gainers);
        model.addAttribute("title", "Top Gainers");
        return "stock-list";
    }

    @GetMapping("/losers")
    public String losers(Model model) {
        List<Stock> stocks = stockRepository.findAll();
        List<Stock> losers = new ArrayList<>();
        for (int i = 0; i < stocks.size(); i++) {
            Stock s = stocks.get(i);
            if (s.getChange() < 0) {
                Stock copy = new Stock();
                copy.setSymbol(s.getSymbol());
                copy.setName(s.getName());
                copy.setPrice(s.getPrice());
                copy.setChange(s.getChange());
                copy.setChangePercent(s.getChangePercent());
                copy.setVolume(s.getVolume());
                copy.setHigh(s.getHigh());
                copy.setLow(s.getLow());
                copy.setOpen(s.getOpen());
                copy.setPreviousClose(s.getPreviousClose());
                losers.add(copy);
            }
        }
        losers.sort((a, b) -> Double.compare(a.getChangePercent(), b.getChangePercent()));
        model.addAttribute("stocks", losers);
        model.addAttribute("title", "Top Losers");
        return "stock-list";
    }

    @GetMapping("/volume")
    public String topVolume(Model model) {
        List<Stock> stocks = stockRepository.findAll();
        List<Stock> result = new ArrayList<>();
        for (int i = 0; i < stocks.size(); i++) {
            Stock s = stocks.get(i);
            Stock copy = new Stock();
            copy.setSymbol(s.getSymbol());
            copy.setName(s.getName());
            copy.setPrice(s.getPrice());
            copy.setChange(s.getChange());
            copy.setChangePercent(s.getChangePercent());
            copy.setVolume(s.getVolume());
            copy.setHigh(s.getHigh());
            copy.setLow(s.getLow());
            copy.setOpen(s.getOpen());
            copy.setPreviousClose(s.getPreviousClose());
            result.add(copy);
        }
        result.sort((a, b) -> Long.compare(b.getVolume(), a.getVolume()));
        model.addAttribute("stocks", result);
        model.addAttribute("title", "Top Volume");
        return "stock-list";
    }
}

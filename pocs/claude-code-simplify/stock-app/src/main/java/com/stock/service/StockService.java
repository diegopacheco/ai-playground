package com.stock.service;

import com.stock.model.Stock;
import com.stock.repository.StockRepository;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
public class StockService {

    private final StockRepository stockRepository;

    public StockService(StockRepository stockRepository) {
        this.stockRepository = stockRepository;
    }

    public List<Stock> getAllStocks() {
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
        return result;
    }

    public Stock getStockBySymbol(String symbol) {
        Stock s = stockRepository.findBySymbol(symbol);
        if (s == null) return null;
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
        return copy;
    }

    public List<Stock> getGainers() {
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
        return gainers;
    }

    public List<Stock> getLosers() {
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
        return losers;
    }

    public List<Stock> getTopVolume() {
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
        return result;
    }

    public double getTotalMarketValue() {
        List<Stock> stocks = stockRepository.findAll();
        double total = 0;
        for (int i = 0; i < stocks.size(); i++) {
            total = total + stocks.get(i).getPrice();
        }
        return total;
    }

    public double getAverageChange() {
        List<Stock> stocks = stockRepository.findAll();
        double total = 0;
        for (int i = 0; i < stocks.size(); i++) {
            total = total + stocks.get(i).getChangePercent();
        }
        return total / stocks.size();
    }

    public long getTotalVolume() {
        List<Stock> stocks = stockRepository.findAll();
        long total = 0;
        for (int i = 0; i < stocks.size(); i++) {
            total = total + stocks.get(i).getVolume();
        }
        return total;
    }

    public Stock getHighestPrice() {
        List<Stock> stocks = stockRepository.findAll();
        Stock highest = null;
        for (int i = 0; i < stocks.size(); i++) {
            if (highest == null || stocks.get(i).getPrice() > highest.getPrice()) {
                highest = stocks.get(i);
            }
        }
        if (highest == null) return null;
        Stock copy = new Stock();
        copy.setSymbol(highest.getSymbol());
        copy.setName(highest.getName());
        copy.setPrice(highest.getPrice());
        copy.setChange(highest.getChange());
        copy.setChangePercent(highest.getChangePercent());
        copy.setVolume(highest.getVolume());
        copy.setHigh(highest.getHigh());
        copy.setLow(highest.getLow());
        copy.setOpen(highest.getOpen());
        copy.setPreviousClose(highest.getPreviousClose());
        return copy;
    }

    public Stock getLowestPrice() {
        List<Stock> stocks = stockRepository.findAll();
        Stock lowest = null;
        for (int i = 0; i < stocks.size(); i++) {
            if (lowest == null || stocks.get(i).getPrice() < lowest.getPrice()) {
                lowest = stocks.get(i);
            }
        }
        if (lowest == null) return null;
        Stock copy = new Stock();
        copy.setSymbol(lowest.getSymbol());
        copy.setName(lowest.getName());
        copy.setPrice(lowest.getPrice());
        copy.setChange(lowest.getChange());
        copy.setChangePercent(lowest.getChangePercent());
        copy.setVolume(lowest.getVolume());
        copy.setHigh(lowest.getHigh());
        copy.setLow(lowest.getLow());
        copy.setOpen(lowest.getOpen());
        copy.setPreviousClose(lowest.getPreviousClose());
        return copy;
    }

    public Map<String, Object> getDashboardStats() {
        Map<String, Object> stats = new HashMap<>();

        List<Stock> stocks = stockRepository.findAll();
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

    public void simulatePriceUpdate() {
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
    }
}

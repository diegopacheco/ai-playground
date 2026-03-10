# /simplify Report - Stock App

## Summary

| Metric | Before | After |
|--------|--------|-------|
| StockService.java | 265 lines | 87 lines |
| DashboardController.java | 167 lines | 65 lines |
| StockApiController.java | 181 lines | 52 lines |
| Stock.java | 52 lines | 67 lines (+copy constructor) |
| Total Java lines | 665 | 271 |
| Copy-paste blocks | 12 | 0 |
| Stats loops (5 separate) | 3 locations | 1 single-pass |
| Service bypasses | 8 methods | 0 |

## Issues Found and Fixed

### 1. Stock.java - Added Copy Constructor

**Before:** No copy constructor, forcing 10-line setter blocks everywhere.
```java
public class Stock {
    public Stock() {}
    public Stock(String symbol, String name, double price, ...) { ... }
}
```

**After:** Added copy constructor.
```java
public Stock(Stock other) {
    this.symbol = other.symbol;
    this.name = other.name;
    this.price = other.price;
    this.change = other.change;
    this.changePercent = other.changePercent;
    this.volume = other.volume;
    this.high = other.high;
    this.low = other.low;
    this.open = other.open;
    this.previousClose = other.previousClose;
}
```

### 2. StockService.java - Eliminated All Duplication (265 -> 87 lines)

**Before:** 10-field copy block repeated 8 times, 5 separate loops for stats, redundant methods, new Random() per call.
```java
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
        if (stocks.get(i).getChange() > 0) { gainerCount++; }
    }
    stats.put("gainerCount", gainerCount);
    int loserCount = 0;
    for (int i = 0; i < stocks.size(); i++) {
        if (stocks.get(i).getChange() < 0) { loserCount++; }
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
```

**After:** Streams, copy constructor, single-pass stats, reusable Random field. Removed redundant getTotalMarketValue(), getAverageChange(), getTotalVolume(), getHighestPrice(), getLowestPrice() methods.
```java
private final Random random = new Random();

public List<Stock> getAllStocks() {
    return stockRepository.findAll();
}

public List<Stock> getGainers() {
    return stockRepository.findAll().stream()
            .filter(s -> s.getChange() > 0)
            .sorted((a, b) -> Double.compare(b.getChangePercent(), a.getChangePercent()))
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
```

### 3. DashboardController.java - Use Service Layer (167 -> 65 lines)

**Before:** Injected both StockService AND StockRepository. Bypassed service for stats, gainers, losers, volume. Reimplemented all logic inline.
```java
public class DashboardController {
    private final StockService stockService;
    private final StockRepository stockRepository;

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
        // ... 30 more lines of duplicated stats loops ...
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
                // ... 8 more setters ...
                gainers.add(copy);
            }
        }
        gainers.sort((a, b) -> Double.compare(b.getChangePercent(), a.getChangePercent()));
        model.addAttribute("stocks", gainers);
        model.addAttribute("title", "Top Gainers");
        return "stock-list";
    }
    // losers() and topVolume() same pattern...
}
```

**After:** Only uses StockService. Each handler is 3-4 lines.
```java
public class DashboardController {
    private final StockService stockService;

    @GetMapping("/")
    public String dashboard(Model model) {
        model.addAttribute("stocks", stockService.getAllStocks());
        Map<String, Object> stats = stockService.getDashboardStats();
        stats.forEach(model::addAttribute);
        return "dashboard";
    }

    @GetMapping("/gainers")
    public String gainers(Model model) {
        model.addAttribute("stocks", stockService.getGainers());
        model.addAttribute("title", "Top Gainers");
        return "stock-list";
    }
}
```

### 4. StockApiController.java - Use Service, Return Stock Directly (181 -> 52 lines)

**Before:** Injected only StockRepository, ignored StockService. Manually converted Stock to Map<String,Object> with 10 map.put() calls (4 times). Duplicated simulate logic.
```java
public class StockApiController {
    private final StockRepository stockRepository;

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

    @PostMapping("/simulate")
    public Map<String, String> simulate() {
        List<Stock> stocks = stockRepository.findAll();
        Random random = new Random();
        for (int i = 0; i < stocks.size(); i++) {
            // ... 15 lines of duplicated simulation logic ...
        }
        // ...
    }
}
```

**After:** Uses StockService. Returns Stock directly (Jackson serializes it). Each handler is 1 line.
```java
public class StockApiController {
    private final StockService stockService;

    @GetMapping("/stocks")
    public List<Stock> getAllStocks() {
        return stockService.getAllStocks();
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
```

## What Was Fixed

- **12 copy-paste blocks eliminated** (10-field setter copy appeared 12 times -> replaced with copy constructor or removed)
- **4 stringly-typed Map conversions eliminated** (Stock returned directly, Jackson handles JSON)
- **3 stats computation duplicates -> 1** (single-pass loop in StockService)
- **5 separate loops -> 1** for dashboard stats
- **8 service bypasses -> 0** (controllers now use StockService exclusively)
- **Redundant Random allocation** -> single field instance
- **5 unused methods removed** (getTotalMarketValue, getAverageChange, getTotalVolume, getHighestPrice, getLowestPrice)
- **Indexed for-loops -> enhanced for-each / streams**
- **Total: ~400 lines removed (665 -> 271)**

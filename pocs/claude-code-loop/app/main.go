package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"sync"
	"time"
)

var globalStocks []Stock
var globalMutex sync.Mutex

type Stock struct {
	Symbol string  `json:"symbol"`
	Price  float64 `json:"price"`
	Change float64 `json:"change"`
}

func init() {
	globalStocks = []Stock{
		{Symbol: "AAPL", Price: 178.50, Change: 0.0},
		{Symbol: "GOOGL", Price: 141.20, Change: 0.0},
		{Symbol: "MSFT", Price: 415.30, Change: 0.0},
		{Symbol: "AMZN", Price: 185.60, Change: 0.0},
		{Symbol: "TSLA", Price: 248.90, Change: 0.0},
		{Symbol: "NVDA", Price: 875.40, Change: 0.0},
		{Symbol: "META", Price: 505.75, Change: 0.0},
		{Symbol: "NFLX", Price: 612.30, Change: 0.0},
	}
}

func simulatePrices() {
	for {
		globalMutex.Lock()
		for i := range globalStocks {
			delta := (rand.Float64() - 0.48) * 5.0
			globalStocks[i].Change = delta
			globalStocks[i].Price += delta
			if globalStocks[i].Price < 1 {
				globalStocks[i].Price = 1
			}
		}
		globalMutex.Unlock()
		time.Sleep(2 * time.Second)
	}
}

func apiHandler(w http.ResponseWriter, r *http.Request) {
	globalMutex.Lock()
	data, _ := json.Marshal(globalStocks)
	globalMutex.Unlock()
	w.Header().Set("Content-Type", "application/json")
	w.Write(data)
}

func dashboardHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	fmt.Fprint(w, `<!DOCTYPE html>
<html>
<head>
<title>Stock Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0e17; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; padding: 20px; }
h1 { text-align: center; color: #00d4ff; margin-bottom: 30px; font-size: 28px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; max-width: 1200px; margin: 0 auto; }
.card { background: #151b2b; border-radius: 12px; padding: 24px; border: 1px solid #1e2a3a; }
.symbol { font-size: 22px; font-weight: bold; color: #ffffff; }
.price { font-size: 36px; font-weight: bold; margin: 12px 0; }
.change { font-size: 16px; font-weight: 600; }
.up { color: #00e676; }
.down { color: #ff1744; }
.bar-container { margin-top: 14px; height: 6px; background: #1e2a3a; border-radius: 3px; overflow: hidden; }
.bar { height: 100%; border-radius: 3px; transition: width 0.5s; }
.timestamp { text-align: center; color: #555; margin-top: 20px; font-size: 13px; }
</style>
</head>
<body>
<h1>Stock Price Dashboard</h1>
<div class="grid" id="grid"></div>
<div class="timestamp" id="ts"></div>
<script>
function render(stocks) {
  var grid = document.getElementById("grid");
  grid.innerHTML = "";
  var maxPrice = 0;
  for (var i = 0; i < stocks.length; i++) {
    if (stocks[i].price > maxPrice) maxPrice = stocks[i].price;
  }
  for (var i = 0; i < stocks.length; i++) {
    var s = stocks[i];
    var isUp = s.change >= 0;
    var sign = isUp ? "+" : "";
    var cls = isUp ? "up" : "down";
    var pct = (maxPrice > 0) ? (s.price / maxPrice * 100) : 0;
    var barColor = isUp ? "#00e676" : "#ff1744";
    var card = document.createElement("div");
    card.className = "card";
    card.innerHTML =
      '<div class="symbol">' + s.symbol + '</div>' +
      '<div class="price ' + cls + '">$' + s.price.toFixed(2) + '</div>' +
      '<div class="change ' + cls + '">' + sign + s.change.toFixed(2) + '</div>' +
      '<div class="bar-container"><div class="bar" style="width:' + pct.toFixed(1) + '%;background:' + barColor + '"></div></div>';
    grid.appendChild(card);
  }
  document.getElementById("ts").textContent = "Last update: " + new Date().toLocaleTimeString();
}
function poll() {
  var xhr = new XMLHttpRequest();
  xhr.open("GET", "/api/stocks");
  xhr.onload = function() {
    if (xhr.status === 200) {
      render(JSON.parse(xhr.responseText));
    }
  };
  xhr.send();
}
poll();
setInterval(poll, 2000);
</script>
</body>
</html>`)
}

func main() {
	go simulatePrices()
	http.HandleFunc("/", dashboardHandler)
	http.HandleFunc("/api/stocks", apiHandler)
	fmt.Println("Server running on http://localhost:8080")
	http.ListenAndServe(":8080", nil)
}

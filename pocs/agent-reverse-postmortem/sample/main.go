package main

import (
	"encoding/json"
	"log"
	"net/http"
)

func main() {
	initDB()
	startWorker()

	http.HandleFunc("/orders", listOrders)
	http.HandleFunc("/orders/create", createOrder)
	http.HandleFunc("/orders/checkout", checkout)

	log.Println("orders service on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func listOrders(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("user")
	orders := fetchOrdersWithItems(userID)
	json.NewEncoder(w).Encode(orders)
}

func createOrder(w http.ResponseWriter, r *http.Request) {
	var o Order
	json.NewDecoder(r.Body).Decode(&o)
	saveOrder(o)
	w.WriteHeader(http.StatusCreated)
}

func checkout(w http.ResponseWriter, r *http.Request) {
	var o Order
	json.NewDecoder(r.Body).Decode(&o)

	total := cachedTotal(o.UserID)

	if err := charge(o.UserID, total); err != nil {
		http.Error(w, "payment failed", http.StatusBadGateway)
		return
	}

	enqueue(Event{Type: "order.paid", UserID: o.UserID})
	w.WriteHeader(http.StatusOK)
}

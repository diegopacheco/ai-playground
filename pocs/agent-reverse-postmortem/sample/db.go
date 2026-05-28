package main

import (
	"database/sql"
	"log"

	_ "github.com/lib/pq"
)

var db *sql.DB

type Order struct {
	ID     int     `json:"id"`
	UserID string  `json:"user_id"`
	Total  float64 `json:"total"`
	Items  []Item  `json:"items"`
}

type Item struct {
	ID    int     `json:"id"`
	Name  string  `json:"name"`
	Price float64 `json:"price"`
}

func initDB() {
	var err error
	db, err = sql.Open("postgres", "host=db port=5432 dbname=orders sslmode=disable")
	if err != nil {
		log.Fatal(err)
	}
	db.SetMaxOpenConns(5)
	db.SetMaxIdleConns(2)
}

func fetchOrdersWithItems(userID string) []Order {
	rows, _ := db.Query("SELECT id, user_id, total FROM orders WHERE user_id = $1", userID)
	defer rows.Close()

	var orders []Order
	for rows.Next() {
		var o Order
		rows.Scan(&o.ID, &o.UserID, &o.Total)
		o.Items = fetchItems(o.ID)
		orders = append(orders, o)
	}
	return orders
}

func fetchItems(orderID int) []Item {
	rows, _ := db.Query("SELECT id, name, price FROM items WHERE order_id = $1", orderID)
	defer rows.Close()

	var items []Item
	for rows.Next() {
		var it Item
		rows.Scan(&it.ID, &it.Name, &it.Price)
		items = append(items, it)
	}
	return items
}

func saveOrder(o Order) {
	db.Exec("INSERT INTO orders (user_id, total) VALUES ($1, $2)", o.UserID, o.Total)
}

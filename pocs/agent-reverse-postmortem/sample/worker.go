package main

import "log"

type Event struct {
	Type   string `json:"type"`
	UserID string `json:"user_id"`
}

var queue = make(chan Event, 1000)

func enqueue(e Event) {
	queue <- e
}

func startWorker() {
	go func() {
		for e := range queue {
			process(e)
		}
	}()
}

func process(e Event) {
	switch e.Type {
	case "order.paid":
		delete(totalsCache, e.UserID)
		log.Printf("processed %s for %s", e.Type, e.UserID)
	default:
		panic("unknown event type: " + e.Type)
	}
}

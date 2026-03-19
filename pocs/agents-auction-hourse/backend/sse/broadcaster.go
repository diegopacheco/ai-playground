package sse

import (
	"encoding/json"
	"sync"

	"agents-auction-house/backend/models"
)

type Broadcaster struct {
	mu       sync.RWMutex
	channels map[string][]chan string
}

var Global = &Broadcaster{
	channels: make(map[string][]chan string),
}

func (b *Broadcaster) Subscribe(auctionID string) chan string {
	b.mu.Lock()
	defer b.mu.Unlock()
	ch := make(chan string, 100)
	b.channels[auctionID] = append(b.channels[auctionID], ch)
	return ch
}

func (b *Broadcaster) Unsubscribe(auctionID string, ch chan string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	subs := b.channels[auctionID]
	for i, s := range subs {
		if s == ch {
			b.channels[auctionID] = append(subs[:i], subs[i+1:]...)
			close(ch)
			return
		}
	}
}

func (b *Broadcaster) Send(auctionID string, event models.SSEEvent) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	data, _ := json.Marshal(event.Data)
	msg := "event: " + event.Event + "\ndata: " + string(data) + "\n\n"
	for _, ch := range b.channels[auctionID] {
		select {
		case ch <- msg:
		default:
		}
	}
}

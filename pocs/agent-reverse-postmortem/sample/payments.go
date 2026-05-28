package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
)

const paymentsURL = "https://payments.internal/charge"

func charge(userID string, amount float64) error {
	body, _ := json.Marshal(map[string]any{"user": userID, "amount": amount})

	for attempt := 0; attempt < 5; attempt++ {
		resp, err := http.Post(paymentsURL, "application/json", bytes.NewReader(body))
		if err != nil {
			continue
		}
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			return nil
		}
	}
	return errors.New("charge failed after retries")
}

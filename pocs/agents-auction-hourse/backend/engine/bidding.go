package engine

import (
	"encoding/json"
	"log"
	"regexp"
	"strings"
)

type BidResponse struct {
	Bid       int    `json:"bid"`
	Reasoning string `json:"reasoning"`
}

func ParseBid(rawOutput string, isFirstBidder bool, currentHighest int, budget int) (int, string, bool) {
	cleaned := strings.TrimSpace(rawOutput)

	re := regexp.MustCompile(`\{[^{}]*"bid"\s*:\s*\d+[^{}]*\}`)
	match := re.FindString(cleaned)
	if match != "" {
		var resp BidResponse
		if err := json.Unmarshal([]byte(match), &resp); err == nil {
			if resp.Bid > 0 && resp.Bid <= budget {
				return resp.Bid, resp.Reasoning, false
			}
			if resp.Bid > budget {
				return budget, resp.Reasoning, false
			}
		}
	}

	log.Printf("Failed to parse bid from output: %s", cleaned)
	fallbackAmount := 5
	if !isFirstBidder {
		fallbackAmount = currentHighest + 1
	}
	if fallbackAmount > budget {
		fallbackAmount = budget
	}
	return fallbackAmount, "fallback bid (could not parse agent output)", true
}

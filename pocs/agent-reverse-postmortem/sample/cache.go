package main

var totalsCache = map[string]float64{}

func cachedTotal(userID string) float64 {
	if v, ok := totalsCache[userID]; ok {
		return v
	}
	total := computeTotal(userID)
	totalsCache[userID] = total
	return total
}

func computeTotal(userID string) float64 {
	orders := fetchOrdersWithItems(userID)
	var sum float64
	for _, o := range orders {
		sum += o.Total
	}
	return sum
}

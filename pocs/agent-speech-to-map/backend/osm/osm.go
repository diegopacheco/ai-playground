package osm

import (
	"context"
	"io"
	"math"
	"net/http"
	"time"
)

const userAgent = "speech-to-map-poc/1.0 (geospatial agent POC)"

var httpClient = &http.Client{Timeout: 15 * time.Second}

type Coord struct {
	Lat float64 `json:"lat"`
	Lon float64 `json:"lon"`
}

type Place struct {
	Name         string  `json:"name"`
	Lat          float64 `json:"lat"`
	Lon          float64 `json:"lon"`
	Address      string  `json:"address"`
	OpeningHours string  `json:"opening_hours"`
	DistanceM    int     `json:"distance_m"`
}

type Route struct {
	To        Coord       `json:"to"`
	Mode      string      `json:"mode"`
	DistanceM int         `json:"distance_m"`
	DurationS int         `json:"duration_s"`
	Geometry  [][]float64 `json:"geometry"`
}

func get(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func haversineM(lat1, lon1, lat2, lon2 float64) int {
	const r = 6371000.0
	p := math.Pi / 180
	a := 0.5 - math.Cos((lat2-lat1)*p)/2 +
		math.Cos(lat1*p)*math.Cos(lat2*p)*(1-math.Cos((lon2-lon1)*p))/2
	return int(2 * r * math.Asin(math.Sqrt(a)))
}

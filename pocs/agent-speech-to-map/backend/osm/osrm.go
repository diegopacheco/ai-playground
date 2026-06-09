package osm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

const osrmBase = "https://router.project-osrm.org/route/v1/"

func GetRoute(ctx context.Context, fromLat, fromLon, toLat, toLon float64, mode string) (*Route, error) {
	profile := normalizeProfile(mode)
	route, err := osrmRequest(ctx, profile, fromLat, fromLon, toLat, toLon, mode)
	if err != nil && profile != "driving" {
		route, err = osrmRequest(ctx, "driving", fromLat, fromLon, toLat, toLon, mode)
	}
	return route, err
}

func osrmRequest(ctx context.Context, profile string, fromLat, fromLon, toLat, toLon float64, mode string) (*Route, error) {
	u := fmt.Sprintf("%s%s/%f,%f;%f,%f?overview=full&geometries=geojson",
		osrmBase, profile, fromLon, fromLat, toLon, toLat)
	body, err := get(ctx, u)
	if err != nil {
		return nil, err
	}
	var parsed struct {
		Code   string `json:"code"`
		Routes []struct {
			Distance float64 `json:"distance"`
			Duration float64 `json:"duration"`
			Geometry struct {
				Coordinates [][]float64 `json:"coordinates"`
			} `json:"geometry"`
		} `json:"routes"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, err
	}
	if parsed.Code != "Ok" || len(parsed.Routes) == 0 {
		return nil, fmt.Errorf("no route found (code %q)", parsed.Code)
	}
	r := parsed.Routes[0]
	geometry := make([][]float64, 0, len(r.Geometry.Coordinates))
	for _, c := range r.Geometry.Coordinates {
		if len(c) >= 2 {
			geometry = append(geometry, []float64{c[1], c[0]})
		}
	}
	durationS := int(r.Duration)
	if normalizeProfile(mode) == "foot" {
		durationS = int(r.Distance / 1.4)
	}
	return &Route{
		To:        Coord{Lat: toLat, Lon: toLon},
		Mode:      mode,
		DistanceM: int(r.Distance),
		DurationS: durationS,
		Geometry:  geometry,
	}, nil
}

func normalizeProfile(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "foot", "walk", "walking":
		return "foot"
	case "bike", "cycling", "bicycle":
		return "bike"
	default:
		return "driving"
	}
}

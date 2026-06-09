package osm

import (
	"context"
	"encoding/json"
	"net/url"
	"strconv"
)

const nominatimURL = "https://nominatim.openstreetmap.org/search"

func Geocode(ctx context.Context, query string) ([]Place, error) {
	u := nominatimURL + "?" + url.Values{
		"q":      {query},
		"format": {"jsonv2"},
		"limit":  {"5"},
	}.Encode()
	body, err := get(ctx, u)
	if err != nil {
		return nil, err
	}
	var raw []struct {
		Lat         string `json:"lat"`
		Lon         string `json:"lon"`
		Name        string `json:"name"`
		DisplayName string `json:"display_name"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, err
	}
	places := make([]Place, 0, len(raw))
	for _, r := range raw {
		lat, err1 := strconv.ParseFloat(r.Lat, 64)
		lon, err2 := strconv.ParseFloat(r.Lon, 64)
		if err1 != nil || err2 != nil {
			continue
		}
		name := r.Name
		if name == "" {
			name = r.DisplayName
		}
		places = append(places, Place{
			Name:    name,
			Lat:     lat,
			Lon:     lon,
			Address: r.DisplayName,
		})
	}
	return places, nil
}

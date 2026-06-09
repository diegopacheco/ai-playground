package osm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
)

const overpassURL = "https://overpass-api.de/api/interpreter"

func FindPOIs(ctx context.Context, lat, lon, radiusM float64, brand, amenity string, limit int) ([]Place, error) {
	if radiusM <= 0 {
		radiusM = 2000
	}
	if radiusM > 20000 {
		radiusM = 20000
	}
	if limit <= 0 {
		limit = 10
	}
	body, err := overpassPost(ctx, buildOverpassQuery(lat, lon, radiusM, brand, amenity))
	if err != nil {
		return nil, err
	}
	var parsed struct {
		Elements []struct {
			Lat    float64           `json:"lat"`
			Lon    float64           `json:"lon"`
			Center *Coord            `json:"center"`
			Tags   map[string]string `json:"tags"`
		} `json:"elements"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, err
	}
	places := make([]Place, 0, len(parsed.Elements))
	for _, e := range parsed.Elements {
		plat, plon := e.Lat, e.Lon
		if e.Center != nil {
			plat, plon = e.Center.Lat, e.Center.Lon
		}
		if plat == 0 && plon == 0 {
			continue
		}
		name := e.Tags["name"]
		if name == "" {
			name = e.Tags["brand"]
		}
		if name == "" {
			continue
		}
		places = append(places, Place{
			Name:         name,
			Lat:          plat,
			Lon:          plon,
			Address:      formatAddress(e.Tags),
			OpeningHours: e.Tags["opening_hours"],
			DistanceM:    haversineM(lat, lon, plat, plon),
		})
	}
	sort.Slice(places, func(i, j int) bool { return places[i].DistanceM < places[j].DistanceM })
	if len(places) > limit {
		places = places[:limit]
	}
	return places, nil
}

func buildOverpassQuery(lat, lon, radiusM float64, brand, amenity string) string {
	area := fmt.Sprintf("(around:%.0f,%f,%f)", radiusM, lat, lon)
	var clauses []string
	add := func(filter string) {
		clauses = append(clauses, fmt.Sprintf("node%s%s;", filter, area))
		clauses = append(clauses, fmt.Sprintf("way%s%s;", filter, area))
	}
	brand = strings.TrimSpace(brand)
	amenity = strings.TrimSpace(amenity)
	switch {
	case brand != "":
		add(fmt.Sprintf(`["name"~"%s",i]`, escapeOverpass(brand)))
	case amenity != "":
		a := escapeOverpass(amenity)
		add(fmt.Sprintf(`["amenity"="%s"]`, a))
		add(fmt.Sprintf(`["shop"="%s"]`, a))
	default:
		add(`["amenity"]`)
	}
	return fmt.Sprintf("[out:json][timeout:25];(%s);out center tags 40;", strings.Join(clauses, ""))
}

func escapeOverpass(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	return strings.ReplaceAll(s, `"`, `\"`)
}

func formatAddress(t map[string]string) string {
	var parts []string
	if hn := t["addr:housenumber"]; hn != "" {
		if st := t["addr:street"]; st != "" {
			parts = append(parts, hn+" "+st)
		} else {
			parts = append(parts, hn)
		}
	} else if st := t["addr:street"]; st != "" {
		parts = append(parts, st)
	}
	if c := t["addr:city"]; c != "" {
		parts = append(parts, c)
	}
	return strings.Join(parts, ", ")
}

func overpassPost(ctx context.Context, query string) ([]byte, error) {
	form := url.Values{"data": {query}}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, overpassURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", userAgent)
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

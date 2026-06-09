package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v2"
	"speechtomap/osm"
)

type toolResult struct {
	content string
	places  []osm.Place
	route   *osm.Route
}

func toolSchemas() []openai.ChatCompletionToolUnionParam {
	return []openai.ChatCompletionToolUnionParam{
		openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        "geocode_place",
			Description: openai.String("Resolve a named place to coordinates using OpenStreetMap Nominatim."),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{"type": "string", "description": "Place to resolve, e.g. 'Times Square, New York'"},
				},
				"required": []string{"query"},
			},
		}),
		openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        "find_pois",
			Description: openai.String("Find points of interest near a coordinate using OpenStreetMap Overpass."),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"lat":      map[string]any{"type": "number"},
					"lon":      map[string]any{"type": "number"},
					"radius_m": map[string]any{"type": "number", "description": "Search radius in meters, default 2000"},
					"brand":    map[string]any{"type": "string", "description": "Optional brand name, e.g. 'Burger King'"},
					"amenity":  map[string]any{"type": "string", "description": "Optional OSM amenity or shop, e.g. 'pharmacy'"},
					"open_now": map[string]any{"type": "boolean", "description": "Optional hint to prefer currently-open places"},
				},
				"required": []string{"lat", "lon"},
			},
		}),
		openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        "get_route",
			Description: openai.String("Route between two coordinates using OpenStreetMap OSRM."),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]any{
					"from_lat": map[string]any{"type": "number"},
					"from_lon": map[string]any{"type": "number"},
					"to_lat":   map[string]any{"type": "number"},
					"to_lon":   map[string]any{"type": "number"},
					"mode":     map[string]any{"type": "string", "enum": []string{"foot", "driving", "cycling"}, "description": "Travel mode, default foot"},
				},
				"required": []string{"from_lat", "from_lon", "to_lat", "to_lon"},
			},
		}),
	}
}

func dispatchTool(ctx context.Context, name, args string) toolResult {
	switch name {
	case "geocode_place":
		var a struct {
			Query string `json:"query"`
		}
		json.Unmarshal([]byte(args), &a)
		places, err := osm.Geocode(ctx, a.Query)
		if err != nil {
			return toolResult{content: toolError(err)}
		}
		return toolResult{content: toJSON(map[string]any{"places": places}), places: places}
	case "find_pois":
		var a struct {
			Lat     float64 `json:"lat"`
			Lon     float64 `json:"lon"`
			RadiusM float64 `json:"radius_m"`
			Brand   string  `json:"brand"`
			Amenity string  `json:"amenity"`
			OpenNow bool    `json:"open_now"`
		}
		json.Unmarshal([]byte(args), &a)
		places, err := osm.FindPOIs(ctx, a.Lat, a.Lon, a.RadiusM, a.Brand, a.Amenity, 10)
		if err != nil {
			return toolResult{content: toolError(err)}
		}
		return toolResult{content: toJSON(map[string]any{"places": places}), places: places}
	case "get_route":
		var a struct {
			FromLat float64 `json:"from_lat"`
			FromLon float64 `json:"from_lon"`
			ToLat   float64 `json:"to_lat"`
			ToLon   float64 `json:"to_lon"`
			Mode    string  `json:"mode"`
		}
		json.Unmarshal([]byte(args), &a)
		if a.Mode == "" {
			a.Mode = "foot"
		}
		route, err := osm.GetRoute(ctx, a.FromLat, a.FromLon, a.ToLat, a.ToLon, a.Mode)
		if err != nil {
			return toolResult{content: toolError(err)}
		}
		return toolResult{content: toJSON(route), route: route}
	default:
		return toolResult{content: toolError(fmt.Errorf("unknown tool %q", name))}
	}
}

func toJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return `{"error":"failed to encode tool result"}`
	}
	return string(b)
}

func toolError(err error) string {
	return toJSON(map[string]string{"error": err.Error()})
}

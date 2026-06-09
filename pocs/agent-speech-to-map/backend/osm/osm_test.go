package osm

import (
	"strings"
	"testing"
)

func TestBuildOverpassQueryBrandIsCaseInsensitive(t *testing.T) {
	q := buildOverpassQuery(40.758, -73.9855, 2000, "Burger King", "")
	if !strings.Contains(q, `["name"~"Burger King",i]`) {
		t.Fatalf("brand search must match name case-insensitively so spoken text matches OSM tags; got %q", q)
	}
	if !strings.Contains(q, "(around:2000,40.758000,-73.985500)") {
		t.Fatalf("query must bound the search to the requested radius and point; got %q", q)
	}
	if strings.Contains(q, `"shop"`) || strings.Contains(q, `"amenity"=`) {
		t.Fatalf("a brand query should not also constrain amenity/shop; got %q", q)
	}
}

func TestBuildOverpassQueryAmenityCoversShopAndAmenity(t *testing.T) {
	q := buildOverpassQuery(0, 0, 1500, "", "pharmacy")
	if !strings.Contains(q, `["amenity"="pharmacy"]`) || !strings.Contains(q, `["shop"="pharmacy"]`) {
		t.Fatalf("amenity search must look under both amenity and shop tags so 'pharmacy' is found either way; got %q", q)
	}
}

func TestBuildOverpassQueryDefaultsToAnyAmenity(t *testing.T) {
	q := buildOverpassQuery(0, 0, 2000, "", "")
	if !strings.Contains(q, `["amenity"]`) {
		t.Fatalf("with no brand or amenity the query should still return nearby amenities; got %q", q)
	}
}

func TestEscapeOverpassNeutralizesQuotes(t *testing.T) {
	if got := escapeOverpass(`a"b\c`); got != `a\"b\\c` {
		t.Fatalf("quotes and backslashes must be escaped to avoid breaking the Overpass filter; got %q", got)
	}
}

func TestFormatAddressJoinsNumberStreetCity(t *testing.T) {
	got := formatAddress(map[string]string{
		"addr:housenumber": "350",
		"addr:street":      "7th Avenue",
		"addr:city":        "New York",
	})
	if got != "350 7th Avenue, New York" {
		t.Fatalf("address shown in the popup must read as number+street, city; got %q", got)
	}
}

func TestNormalizeProfileMapsWalkableToFoot(t *testing.T) {
	for _, mode := range []string{"foot", "walk", "walking", "Walking"} {
		if got := normalizeProfile(mode); got != "foot" {
			t.Fatalf("a walkable request must route on the foot profile; mode %q gave %q", mode, got)
		}
	}
	if got := normalizeProfile("cycling"); got != "bike" {
		t.Fatalf("cycling must map to the bike profile; got %q", got)
	}
	if got := normalizeProfile("anything-else"); got != "driving" {
		t.Fatalf("unknown modes must fall back to driving; got %q", got)
	}
}

func TestHaversineApproximatesKnownDistance(t *testing.T) {
	d := haversineM(0, 0, 0, 1)
	if d < 110000 || d > 112000 {
		t.Fatalf("one degree of longitude at the equator is about 111 km; got %d m", d)
	}
}

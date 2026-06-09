package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"speechtomap/osm"
)

type queryRequest struct {
	Text string  `json:"text"`
	Lat  float64 `json:"lat"`
	Lon  float64 `json:"lon"`
	Now  string  `json:"now"`
}

type queryResponse struct {
	Answer string      `json:"answer"`
	Center osm.Coord   `json:"center"`
	Places []osm.Place `json:"places"`
	Route  *osm.Route  `json:"route,omitempty"`
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is not set")
	}
	client := openai.NewClient(option.WithAPIKey(apiKey))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/api/health", handleHealth)
	mux.HandleFunc("/api/query", handleQuery(client))

	log.Printf("backend listening on :%s", port)
	if err := http.ListenAndServe(":"+port, withCORS(mux)); err != nil {
		log.Fatal(err)
	}
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func handleQuery(client openai.Client) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
			return
		}
		var req queryRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
			return
		}
		if strings.TrimSpace(req.Text) == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "text is required"})
			return
		}
		if req.Now == "" {
			req.Now = time.Now().Format(time.RFC3339)
		}

		ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
		defer cancel()

		result, err := runAgent(ctx, client, req)
		if err != nil {
			log.Printf("agent error: %v", err)
			writeJSON(w, http.StatusBadGateway, map[string]string{"error": "agent failed: " + err.Error()})
			return
		}

		places := result.Places
		if places == nil {
			places = []osm.Place{}
		}
		writeJSON(w, http.StatusOK, queryResponse{
			Answer: result.Answer,
			Center: osm.Coord{Lat: req.Lat, Lon: req.Lon},
			Places: places,
			Route:  result.Route,
		})
	}
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

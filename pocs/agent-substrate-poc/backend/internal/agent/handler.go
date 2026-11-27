package agent

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
)

type noteRequest struct {
	Text string `json:"text"`
}

func Routes(m *Memory, logger *slog.Logger) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("GET /state", func(w http.ResponseWriter, r *http.Request) {
		state, err := m.Snapshot()
		if err != nil {
			logger.ErrorContext(r.Context(), "snapshot failed", "err", err)
			http.Error(w, "internal error", http.StatusInternalServerError)
			return
		}
		writeJSON(w, r, logger, http.StatusOK, state)
	})
	mux.HandleFunc("POST /notes", func(w http.ResponseWriter, r *http.Request) {
		var req noteRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096)).Decode(&req); err != nil {
			http.Error(w, "invalid json", http.StatusBadRequest)
			return
		}
		note, err := m.Remember(req.Text)
		switch {
		case errors.Is(err, ErrEmptyNote):
			http.Error(w, err.Error(), http.StatusBadRequest)
		case err != nil:
			logger.ErrorContext(r.Context(), "remember failed", "err", err)
			http.Error(w, "internal error", http.StatusInternalServerError)
		default:
			logger.InfoContext(r.Context(), "note remembered", "id", note.ID)
			writeJSON(w, r, logger, http.StatusCreated, note)
		}
	})
	return mux
}

func writeJSON(w http.ResponseWriter, r *http.Request, logger *slog.Logger, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		logger.ErrorContext(r.Context(), "encode response", "err", err)
	}
}

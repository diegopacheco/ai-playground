package actorclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"agentsubstratepoc/internal/agent"
	"agentsubstratepoc/internal/logx"
)

type Client struct {
	http *http.Client
	host string
}

func New(host string, timeout time.Duration) *Client {
	return &Client{http: &http.Client{Timeout: timeout}, host: host}
}

func (c *Client) State(ctx context.Context, port int) (agent.State, error) {
	var s agent.State
	err := c.do(ctx, http.MethodGet, port, "/state", nil, http.StatusOK, &s)
	return s, err
}

func (c *Client) AddNote(ctx context.Context, port int, text string) (agent.Note, error) {
	body, err := json.Marshal(map[string]string{"text": text})
	if err != nil {
		return agent.Note{}, fmt.Errorf("encode note: %w", err)
	}
	var n agent.Note
	err = c.do(ctx, http.MethodPost, port, "/notes", body, http.StatusCreated, &n)
	return n, err
}

func (c *Client) do(ctx context.Context, method string, port int, path string, body []byte, want int, out any) error {
	url := "http://" + c.host + ":" + strconv.Itoa(port) + path
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build request %s: %w", path, err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(logx.Header, logx.CorrelationID(ctx))
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("%s %s: %w", method, path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != want {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("%s %s: status %d: %s", method, path, resp.StatusCode, bytes.TrimSpace(msg))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

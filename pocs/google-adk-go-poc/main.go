package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"log"
	"net/http"
	"os"
	"strings"

	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type OpenAIModel struct {
	name   string
	apiKey string
}

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIRequest struct {
	Model    string          `json:"model"`
	Messages []openAIMessage `json:"messages"`
}

type openAIChoice struct {
	Message openAIMessage `json:"message"`
}

type openAIError struct {
	Message string `json:"message"`
}

type openAIResponse struct {
	Choices []openAIChoice `json:"choices"`
	Error   *openAIError   `json:"error,omitempty"`
}

func extractText(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var parts []string
	for _, p := range content.Parts {
		if p.Text != "" {
			parts = append(parts, p.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func (m *OpenAIModel) Name() string {
	return m.name
}

func (m *OpenAIModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		var messages []openAIMessage

		if req.Config != nil && req.Config.SystemInstruction != nil {
			text := extractText(req.Config.SystemInstruction)
			if text != "" {
				messages = append(messages, openAIMessage{Role: "system", Content: text})
			}
		}

		for _, c := range req.Contents {
			role := "user"
			if c.Role == "model" {
				role = "assistant"
			}
			text := extractText(c)
			if text != "" {
				messages = append(messages, openAIMessage{Role: role, Content: text})
			}
		}

		body, err := json.Marshal(openAIRequest{Model: m.name, Messages: messages})
		if err != nil {
			yield(nil, fmt.Errorf("marshal request: %w", err))
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
		if err != nil {
			yield(nil, fmt.Errorf("create request: %w", err))
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)

		resp, err := http.DefaultClient.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("do request: %w", err))
			return
		}
		defer resp.Body.Close()

		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			yield(nil, fmt.Errorf("read response: %w", err))
			return
		}

		var result openAIResponse
		if err := json.Unmarshal(respBody, &result); err != nil {
			yield(nil, fmt.Errorf("unmarshal response: %w", err))
			return
		}

		if result.Error != nil {
			yield(nil, fmt.Errorf("openai: %s", result.Error.Message))
			return
		}

		if len(result.Choices) == 0 {
			yield(nil, fmt.Errorf("no choices in response"))
			return
		}

		content := &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{Text: result.Choices[0].Message.Content}},
		}

		yield(&model.LLMResponse{
			Content:      content,
			TurnComplete: true,
		}, nil)
	}
}

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	openaiModel := &OpenAIModel{name: "gpt-4o", apiKey: apiKey}

	myAgent := &llmagent.Agent{
		Name:        "my_agent",
		Model:       openaiModel,
		Instruction: "You are a helpful assistant. Answer user questions clearly and concisely.",
	}

	launcher.New(full.Options{}, "my_agent", myAgent).Start(ctx)
}

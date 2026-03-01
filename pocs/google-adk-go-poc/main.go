package main

import (
	"bufio"
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

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
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

	myAgent, err := llmagent.New(llmagent.Config{
		Name:        "my_agent",
		Model:       openaiModel,
		Instruction: "You are a helpful assistant. Answer user questions clearly and concisely.",
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	sessionService := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:        "openai_adk_app",
		Agent:          myAgent,
		SessionService: sessionService,
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	userID := "user1"
	createResp, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: "openai_adk_app",
		UserID:  userID,
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}
	sess := createResp.Session

	reader := bufio.NewReader(os.Stdin)
	fmt.Println("Agent started. Type your message (Ctrl+C to quit):")

	for {
		fmt.Print("\nYou -> ")
		userInput, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		userInput = strings.TrimSpace(userInput)
		if userInput == "" {
			continue
		}

		userMsg := genai.NewContentFromText(userInput, "user")
		fmt.Print("Agent -> ")
		for event, err := range r.Run(ctx, userID, sess.ID(), userMsg, agent.RunConfig{}) {
			if err != nil {
				fmt.Printf("\nERROR: %v\n", err)
			} else {
				if event.LLMResponse.Content == nil {
					continue
				}
				for _, p := range event.LLMResponse.Content.Parts {
					fmt.Print(p.Text)
				}
			}
		}
		fmt.Println()
	}
}

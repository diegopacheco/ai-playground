package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

type OpenAIRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OpenAIResponse struct {
	Choices []Choice  `json:"choices"`
	Usage   Usage     `json:"usage"`
	Error   *APIError `json:"error,omitempty"`
}

type Choice struct {
	Message Message `json:"message"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

type Agent struct {
	apiKey       string
	conversation []Message
}

const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

func NewAgent() *Agent {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: OPENAI_API_KEY environment variable not set")
		os.Exit(1)
	}

	return &Agent{
		apiKey: apiKey,
		conversation: []Message{
			{
				Role:    "system",
				Content: "You are a helpful coding assistant. Provide clear code examples and explanations.",
			},
		},
	}
}

func (a *Agent) callOpenAI(prompt string) (string, error) {
	a.conversation = append(a.conversation, Message{
		Role:    "user",
		Content: prompt,
	})

	reqBody := OpenAIRequest{
		Model:    "gpt-4",
		Messages: a.conversation,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("JSON marshal error: %v", err)
	}

	req, err := http.NewRequest("POST", OPENAI_API_URL, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("request creation error: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+a.apiKey)

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("HTTP request error: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API returned status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("response read error: %v", err)
	}

	var openAIResp OpenAIResponse
	if err := json.Unmarshal(body, &openAIResp); err != nil {
		return "", fmt.Errorf("JSON unmarshal error: %v", err)
	}

	if openAIResp.Error != nil {
		return "", fmt.Errorf("OpenAI API error: %s", openAIResp.Error.Message)
	}

	if len(openAIResp.Choices) == 0 {
		return "", fmt.Errorf("no response from OpenAI")
	}

	response := openAIResp.Choices[0].Message.Content
	a.conversation = append(a.conversation, Message{
		Role:    "assistant",
		Content: response,
	})

	return response, nil
}

func (a *Agent) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method == "POST" {
		prompt := r.FormValue("prompt")
		if prompt == "" {
			http.Error(w, "Empty prompt", http.StatusBadRequest)
			return
		}

		response, err := a.callOpenAI(prompt)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"response": response})
		return
	}

	tmpl := `<!DOCTYPE html>
<html>
<head>
    <title>Diego Code - AI Assistant</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0a; color: #ffffff; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #10a37f; font-size: 2.5rem; margin-bottom: 10px; }
        .header p { color: #8e8ea0; font-size: 1.1rem; }
        .chat-container { background: #1a1a1a; border-radius: 12px; padding: 0; min-height: 600px; display: flex; flex-direction: column; }
        .messages { flex: 1; padding: 20px; overflow-y: auto; max-height: 500px; }
        .message { margin-bottom: 20px; }
        .message.user { text-align: right; }
        .message.assistant { text-align: left; }
        .message-content { display: inline-block; max-width: 70%; padding: 15px 20px; border-radius: 18px; }
        .message.user .message-content { background: #10a37f; color: white; }
        .message.assistant .message-content { background: #2d2d30; color: #ffffff; }
        .input-area { border-top: 1px solid #2d2d30; padding: 20px; display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 15px 20px; border: 1px solid #2d2d30; border-radius: 25px; background: #0a0a0a; color: #ffffff; font-size: 16px; }
        .input-area input:focus { outline: none; border-color: #10a37f; }
        .input-area button { padding: 15px 30px; background: #10a37f; color: white; border: none; border-radius: 25px; cursor: pointer; font-size: 16px; font-weight: 600; }
        .input-area button:hover { background: #0d8f6f; }
        .input-area button:disabled { background: #2d2d30; cursor: not-allowed; }
        pre { background: #000; padding: 15px; border-radius: 8px; overflow-x: auto; margin: 10px 0; }
        code { background: #2d2d30; padding: 2px 6px; border-radius: 4px; }
        .loading { color: #8e8ea0; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Diego Code</h1>
            <p>Your AI Coding Assistant</p>
        </div>
        
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="message-content">
                        Hello! I'm Diego Code, your AI coding assistant. I can help you with:
                        <br><br>
                        • Writing code in any programming language
                        <br>• Debugging and fixing issues
                        <br>• Explaining code concepts
                        <br>• Code reviews and optimization
                        <br>• Architecture and design patterns
                        <br><br>
                        What would you like to work on today?
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <input type="text" id="promptInput" placeholder="Ask me anything about coding..." autofocus>
                <button onclick="sendMessage()" id="sendBtn">Send</button>
            </div>
        </div>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const promptInput = document.getElementById('promptInput');
        const sendBtn = document.getElementById('sendBtn');

        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + (isUser ? 'user' : 'assistant');
            messageDiv.innerHTML = '<div class="message-content">' + content + '</div>';
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function formatResponse(text) {
            // Simple markdown-like formatting
            return text
                .replace(/` + "`" + `{3}([^` + "`" + `]+)` + "`" + `{3}/g, '<pre>$1</pre>')
                .replace(/` + "`" + `([^` + "`" + `]+)` + "`" + `/g, '<code>$1</code>')
                .replace(/\n/g, '<br>');
        }

        async function sendMessage() {
            const prompt = promptInput.value.trim();
            if (!prompt) return;

            addMessage(prompt, true);
            promptInput.value = '';
            sendBtn.disabled = true;
            sendBtn.textContent = 'Thinking...';

            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message assistant';
            loadingDiv.innerHTML = '<div class="message-content loading">Diego Code is thinking...</div>';
            messagesDiv.appendChild(loadingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            try {
                const formData = new FormData();
                formData.append('prompt', prompt);

                const response = await fetch('/', {
                    method: 'POST',
                    body: formData
                });

                messagesDiv.removeChild(loadingDiv);

                if (response.ok) {
                    const data = await response.json();
                    addMessage(formatResponse(data.response));
                } else {
                    addMessage('Sorry, I encountered an error. Please try again.');
                }
            } catch (error) {
                messagesDiv.removeChild(loadingDiv);
                addMessage('Connection error. Please check your internet connection.');
            }

            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
            promptInput.focus();
        }

        promptInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Focus input on load
        promptInput.focus();
    </script>
</body>
</html>`

	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(tmpl))
}

func (a *Agent) runCLI() {
	fmt.Println("Diego Code - AI Coding Assistant")
	fmt.Println("Type your coding questions or 'quit' to exit")
	fmt.Println("=====================================")

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "quit" || input == "exit" {
			fmt.Println("Goodbye!")
			break
		}

		if input == "" {
			continue
		}

		fmt.Println("\nDiego Code is thinking...")

		response, err := a.callOpenAI(input)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		fmt.Printf("\nDiego Code:\n%s\n", response)
		fmt.Println(strings.Repeat("-", 50))
	}
}

func main() {
	agent := NewAgent()
	for _, arg := range os.Args[1:] {
		if arg == "--web" || arg == "-w" {
			fmt.Println("Starting Diego Code web interface...")
			fmt.Println("Open http://localhost:8080 in your browser")

			http.HandleFunc("/", agent.handleChat)

			if err := http.ListenAndServe(":8080", nil); err != nil {
				fmt.Printf("Server error: %v\n", err)
				os.Exit(1)
			}
			return
		}
	}
	agent.runCLI()
}

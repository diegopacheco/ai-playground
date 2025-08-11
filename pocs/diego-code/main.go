package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
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
	workingDir   string
	totalTokens  int
}

const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

var (
	app        *tview.Application
	inputField *tview.InputField
	chatList   *tview.List
	statusView *tview.TextView
	clockView  *tview.TextView
	agent      *Agent
)

func NewAgent() *Agent {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: OPENAI_API_KEY environment variable not set")
		os.Exit(1)
	}

	workingDir, _ := os.Getwd()

	return &Agent{
		apiKey:     apiKey,
		workingDir: workingDir,
		conversation: []Message{
			{
				Role: "system",
				Content: `You are Diego Code, an AI coding assistant that can create, read, and execute code files. 

When a user asks you to create code:
1. First explain what you are doing
2. Use this EXACT format to create files: <CREATE_FILE:filename.ext>actual code without markdown</CREATE_FILE>
3. Use this EXACT format to run code: <RUN_CODE:filename.ext></RUN_CODE>

When a user asks to see/read/show a file:
4. Use this EXACT format to read files: <READ_FILE:filename.ext></READ_FILE>

IMPORTANT: 
- Put the raw code directly inside the tags, NO markdown code blocks (no backticks)
- Always follow CREATE_FILE with RUN_CODE to execute the program
- Use READ_FILE when users ask to see file contents
- Be precise with the format

Examples:
I will create a Python hello world program for you.
<CREATE_FILE:hello.py>print("Hello, World!")</CREATE_FILE>
<RUN_CODE:hello.py></RUN_CODE>

To show you the contents of main.go:
<READ_FILE:main.go></READ_FILE>

You can create and run files in Python, JavaScript, Go, Java, C++, etc.`,
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
	a.totalTokens += openAIResp.Usage.TotalTokens

	a.conversation = append(a.conversation, Message{
		Role:    "assistant",
		Content: response,
	})

	return response, nil
}

func (a *Agent) createFile(filename, content string) string {
	if !filepath.IsAbs(filename) {
		filename = filepath.Join(a.workingDir, filename)
	}

	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Sprintf("❌ Error creating directory: %v", err)
	}

	if err := os.WriteFile(filename, []byte(content), 0644); err != nil {
		return fmt.Sprintf("❌ Error creating file %s: %v", filename, err)
	}

	return fmt.Sprintf("✅ Created file: %s", filepath.Base(filename))
}

func (a *Agent) runCode(filename, args string) string {
	if !filepath.IsAbs(filename) {
		filename = filepath.Join(a.workingDir, filename)
	}

	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return fmt.Sprintf("❌ File %s does not exist", filename)
	}

	ext := filepath.Ext(filename)
	var cmd *exec.Cmd
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	switch ext {
	case ".py":
		if args != "" {
			cmd = exec.CommandContext(ctx, "python3", filename, args)
		} else {
			cmd = exec.CommandContext(ctx, "python3", filename)
		}
	case ".js":
		if args != "" {
			cmd = exec.CommandContext(ctx, "node", filename, args)
		} else {
			cmd = exec.CommandContext(ctx, "node", filename)
		}
	case ".go":
		if args != "" {
			cmd = exec.CommandContext(ctx, "go", "run", filename, args)
		} else {
			cmd = exec.CommandContext(ctx, "go", "run", filename)
		}
	case ".java":
		className := strings.TrimSuffix(filepath.Base(filename), ".java")
		compileCmd := exec.CommandContext(ctx, "javac", filename)
		if err := compileCmd.Run(); err != nil {
			return fmt.Sprintf("❌ Java compilation failed: %v", err)
		}
		if args != "" {
			cmd = exec.CommandContext(ctx, "java", "-cp", filepath.Dir(filename), className, args)
		} else {
			cmd = exec.CommandContext(ctx, "java", "-cp", filepath.Dir(filename), className)
		}
	case ".cpp", ".cc":
		exeName := strings.TrimSuffix(filename, ext)
		compileCmd := exec.CommandContext(ctx, "g++", filename, "-o", exeName)
		if err := compileCmd.Run(); err != nil {
			return fmt.Sprintf("❌ C++ compilation failed: %v", err)
		}
		if args != "" {
			cmd = exec.CommandContext(ctx, exeName, args)
		} else {
			cmd = exec.CommandContext(ctx, exeName)
		}
	default:
		return fmt.Sprintf("⚠️ Unsupported file type: %s", ext)
	}

	cmd.Dir = a.workingDir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Sprintf("❌ Execution failed: %v\nOutput: %s", err, string(output))
	}

	return fmt.Sprintf("🚀 Executed %s:\n%s", filepath.Base(filename), string(output))
}

func (a *Agent) readFile(filename string) string {
	if !filepath.IsAbs(filename) {
		filename = filepath.Join(a.workingDir, filename)
	}

	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return fmt.Sprintf("❌ File %s does not exist", filename)
	}

	content, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Sprintf("❌ Error reading file %s: %v", filename, err)
	}

	return fmt.Sprintf("📄 Contents of %s:\n```\n%s\n```", filepath.Base(filename), string(content))
}

func (a *Agent) processResponse(response string) string {
	var results []string

	createFileRegex := regexp.MustCompile(`(?s)<CREATE_FILE:([^>]+)>(.*?)</CREATE_FILE>`)
	matches := createFileRegex.FindAllStringSubmatch(response, -1)

	for _, match := range matches {
		if len(match) == 3 {
			filename := strings.TrimSpace(match[1])
			content := strings.TrimSpace(match[2])
			content = strings.TrimPrefix(content, "```python")
			content = strings.TrimPrefix(content, "```javascript")
			content = strings.TrimPrefix(content, "```go")
			content = strings.TrimPrefix(content, "```java")
			content = strings.TrimPrefix(content, "```cpp")
			content = strings.TrimPrefix(content, "```")
			content = strings.TrimSuffix(content, "```")
			content = strings.TrimSpace(content)

			result := a.createFile(filename, content)
			results = append(results, result)
		}
	}
	runCodeRegex := regexp.MustCompile(`(?s)<RUN_CODE:([^>]+)>(.*?)</RUN_CODE>`)
	runMatches := runCodeRegex.FindAllStringSubmatch(response, -1)

	for _, match := range runMatches {
		if len(match) >= 2 {
			filename := strings.TrimSpace(match[1])
			args := ""
			if len(match) > 2 {
				args = strings.TrimSpace(match[2])
			}

			result := a.runCode(filename, args)
			results = append(results, result)
		}
	}
	cleanResponse := createFileRegex.ReplaceAllString(response, "")
	cleanResponse = runCodeRegex.ReplaceAllString(cleanResponse, "")
	if len(results) > 0 {
		cleanResponse += "\n\n" + strings.Join(results, "\n")
	}

	return cleanResponse
}

func addChatMessage(sender, message string) {
	timestamp := time.Now().Format("15:04:05")
	var coloredSender string
	if sender == "You" {
		coloredSender = fmt.Sprintf("[blue]%s[white]", sender)
	} else {
		coloredSender = fmt.Sprintf("[green]%s[white]", sender)
	}

	wrappedMessage := wrapText(message, 80) // 80 chars per line
	chatText := fmt.Sprintf("[gray]%s[white] %s: %s", timestamp, coloredSender, wrappedMessage)

	chatList.AddItem(chatText, "", 0, nil)
	chatList.SetCurrentItem(-1) // Auto-scroll to bottom
}

func wrapText(text string, width int) string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return text
	}

	lines := []string{}
	currentLine := ""

	for _, word := range words {
		if len(currentLine)+len(word)+1 > width && currentLine != "" {
			lines = append(lines, currentLine)
			currentLine = word
		} else {
			if currentLine == "" {
				currentLine = word
			} else {
				currentLine += " " + word
			}
		}
	}
	if currentLine != "" {
		lines = append(lines, currentLine)
	}

	return strings.Join(lines, "\n")
}

func updateStatus(message string) {
	status := fmt.Sprintf("Status: %s | Tokens: %d | Dir: %s", message, agent.totalTokens, agent.workingDir)
	statusView.SetText(status)
}

func updateClock() {
	for {
		now := time.Now()
		clockText := fmt.Sprintf("🕐 %s | %s",
			now.Format("15:04:05"),
			now.Format("Mon Jan 02, 2006"))

		app.QueueUpdateDraw(func() {
			clockView.SetText(clockText)
		})

		time.Sleep(1 * time.Second)
	}
}

func handleSubmit() {
	prompt := strings.TrimSpace(inputField.GetText())
	if prompt == "" {
		return
	}

	inputField.SetText("")
	addChatMessage("You", prompt)

	updateStatus("Diego Code is thinking...")
	addChatMessage("Diego Code", "🤔 Thinking...")

	go func() {
		response, err := agent.callOpenAI(prompt)

		app.QueueUpdateDraw(func() {
			chatList.RemoveItem(chatList.GetItemCount() - 1)

			if err != nil {
				addChatMessage("Diego Code", fmt.Sprintf("❌ Error: %v", err))
				updateStatus("Error occurred")
			} else {
				processedResponse := agent.processResponse(response)
				addChatMessage("Diego Code", processedResponse)
				updateStatus("Ready")
			}
		})
	}()
}

func main() {
	agent = NewAgent()
	for _, arg := range os.Args[1:] {
		if arg == "--cli" || arg == "-c" {
			runCLI()
			return
		}
	}
	app = tview.NewApplication()
	clockView = tview.NewTextView().
		SetTextAlign(tview.AlignCenter).
		SetDynamicColors(true)
	clockView.SetBorder(true).SetTitle("Diego Code - AI Assistant")

	inputField = tview.NewInputField().
		SetLabel("Prompt: ").
		SetFieldWidth(0).
		SetPlaceholder("Type your coding question here...")
	inputField.SetBorder(true).SetTitle("Your Question")

	chatList = tview.NewList().
		ShowSecondaryText(false).
		SetHighlightFullLine(true).
		SetSelectedTextColor(tcell.ColorYellow).
		SetSelectedBackgroundColor(tcell.ColorDarkBlue)
	chatList.SetBorder(true).SetTitle("Chat History")

	statusView = tview.NewTextView().
		SetTextAlign(tview.AlignCenter).
		SetDynamicColors(true)
	statusView.SetBorder(true).SetTitle("Status")

	addChatMessage("Diego Code", `Welcome! I'm your AI coding assistant. I can help you with:
• Writing code in any programming language
• Creating and running programs automatically  
• Debugging and fixing code issues
• Explaining programming concepts

Examples you can try:
• "create a hello world in python and run it"
• "write a fibonacci function in go"
• "create a simple web server in javascript"

Press Enter to send, Tab to switch focus, Ctrl+C to quit.`)

	inputField.SetDoneFunc(func(key tcell.Key) {
		if key == tcell.KeyEnter {
			handleSubmit()
		}
	})

	inputField.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		if event.Key() == tcell.KeyTab {
			app.SetFocus(chatList)
			return nil
		}
		return event
	})

	chatList.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		if event.Key() == tcell.KeyTab {
			app.SetFocus(inputField)
			return nil
		}
		return event
	})

	flex := tview.NewFlex().
		SetDirection(tview.FlexRow).
		AddItem(clockView, 3, 0, false).
		AddItem(inputField, 3, 0, true).
		AddItem(chatList, 0, 1, false).
		AddItem(statusView, 3, 0, false)

	app.SetRoot(flex, true).SetFocus(inputField)

	updateStatus("Ready - Welcome to Diego Code!")
	go updateClock()

	if err := app.Run(); err != nil {
		fmt.Printf("TUI failed to start: %v\n", err)
		fmt.Println("Falling back to CLI mode...")
		runCLI()
	}
}

func runCLI() {
	fmt.Printf("Diego Code - AI Coding Assistant (CLI Mode) | %s\n", time.Now().Format("Mon Jan 02, 2006 15:04:05"))
	fmt.Println("Type your coding questions or 'quit' to exit")
	fmt.Println("=====================================")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		currentTime := time.Now().Format("15:04:05")
		fmt.Printf("\n[%s] > ", currentTime)

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

		response, err := agent.callOpenAI(input)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		processedResponse := agent.processResponse(response)

		fmt.Printf("\nDiego Code:\n%s\n", processedResponse)
		fmt.Println(strings.Repeat("-", 50))
	}
}

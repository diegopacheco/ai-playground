# Claude Code (Rust)

A Rust implementation of an AI coding assistant that uses OpenAI's API with tool calling capabilities.

## OpenAI

This application uses the **GPT-4o** model (`gpt-4o`) via the OpenAI Chat Completions API.

| Setting | Value |
|---------|-------|
| Model | `gpt-4o` |
| API Endpoint | `https://api.openai.com/v1/chat/completions` |
| Tool Choice | `auto` |

The model automatically decides when to use tools based on the user's request. It supports multi-turn conversations with full message history.

## Architecture

```
src/
├── main.rs                 # Application entry point, REPL loop
├── agent.rs                # Agent loop orchestration, tool execution
├── llm/
│   ├── mod.rs              # Module exports
│   ├── types.rs            # Message, ToolCall, ApiResponse structs
│   └── api.rs              # OpenAI API communication
└── tools/
    ├── mod.rs              # Tool registry and dispatcher
    ├── read_file.rs        # File reading tool
    ├── list_files.rs       # Directory listing tool
    ├── edit_file.rs        # File writing tool
    ├── execute_command.rs  # Program execution tool
    └── web_search.rs       # Web page text extraction tool

tests/
└── integration_test.rs     # End-to-end integration tests
```

| Component | Purpose |
|-----------|---------|
| `main.rs` | Initializes the agent, manages REPL loop, handles user input |
| `agent.rs` | Orchestrates LLM calls and tool execution in a loop |
| `llm/types.rs` | Defines data structures for API communication |
| `llm/api.rs` | Makes HTTP requests to OpenAI API |
| `tools/mod.rs` | Registers tools and dispatches execution |

## Tools

### read_file
Reads the contents of a file at the specified path.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | The path to the file to read |

### list_files
Lists files and directories at the specified path.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | no | The path to list (defaults to current directory) |

### edit_file
Creates or overwrites a file with the provided content. Creates parent directories if needed.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | yes | The path where the file should be created/modified |
| `content` | string | yes | The content to write to the file |

### execute_command
Executes a program with arguments. Use this to run commands like `node hello.js`, `python3 main.py`, `java -jar app.jar`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `program` | string | yes | The program to execute (e.g., `node`, `python3`, `java`) |
| `args` | array[string] | no | Array of arguments to pass to the program |

### web_search
Fetches a webpage and extracts its text content, stripping all JavaScript, CSS, and HTML tags.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `url` | string | yes | The URL of the webpage to fetch and extract text from |

## Build

```bash
cargo build
```

## Run

```bash
export OPENAI_API_KEY=your_api_key_here
cargo run
```

## Results

Generating a Python hello world application:

```
---------------------------------------------------
Claude Code (Rust) - Type 'quit' or 'exit' to exit
---------------------------------------------------
> Create a python hello world app called main.py
[Tool: edit_file]
Result: File 'main.py' written successfully

I've created a Python hello world application called `main.py`. The file contains:

print("Hello, World!")

You can run it with: python3 main.py

> Run the python app main.py
[Tool: execute_command]
Result: Hello, World!

The Python script ran successfully and printed "Hello, World!" to the console.

> quit
```

## Tests

### Test Summary

| Category | Count | Description |
|----------|-------|-------------|
| Unit Tests | 94 | Tests for all modules |
| Integration Tests | 5 | Application startup, exit, API key validation |
| E2E Tests (ignored) | 3 | Require real OPENAI_API_KEY |
| **Total** | **102** | |

### Unit Tests by Module

| Module | Tests | Coverage |
|--------|-------|----------|
| `tools/read_file.rs` | 5 | File reading, error handling, edge cases |
| `tools/list_files.rs` | 6 | Directory listing, JSON output, empty dirs |
| `tools/edit_file.rs` | 6 | File creation, overwrite, parent dirs |
| `tools/execute_command.rs` | 11 | Program execution, args, error handling |
| `tools/mod.rs` | 19 | Tool registry, dispatcher, all tool routing |
| `tools/web_search.rs` | 13 | HTML parsing, text extraction, content filtering |
| `llm/types.rs` | 16 | Serialization, deserialization of all types |
| `llm/api.rs` | 10 | Request body construction, constants |
| `agent.rs` | 17 | Message handling, truncation, tool processing |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_application_starts_and_shows_banner` | Verifies app starts and displays banner |
| `test_application_exits_on_quit` | Verifies quit command works |
| `test_application_requires_api_key` | Verifies error when OPENAI_API_KEY missing |
| `test_tools_are_available` | Verifies all tool files exist |
| `test_project_compiles` | Verifies cargo check passes |

### E2E Tests (Ignored by Default)

| Test | Description |
|------|-------------|
| `test_end_to_end_create_file` | Creates a file using the LLM |
| `test_end_to_end_list_files` | Lists files using the LLM |
| `test_end_to_end_execute_command` | Executes a command using the LLM |

### Running Tests

```bash
cargo test
```

```
running 94 tests
...
test result: ok. 94 passed; 0 failed; 0 ignored

running 8 tests
...
test result: ok. 5 passed; 0 failed; 3 ignored
```

Run E2E tests (requires OPENAI_API_KEY):
```bash
export OPENAI_API_KEY=your_api_key_here
cargo test -- --ignored
```

Run a specific test:
```bash
cargo test test_read_file_success
```

Run tests with output:
```bash
cargo test -- --nocapture
```

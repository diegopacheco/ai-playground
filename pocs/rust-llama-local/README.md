### Setup

Download a Llama 3 GGUF model:
```bash
mkdir -p models
cd models
curl -L -O https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
mv Meta-Llama-3-8B-Instruct.Q4_K_M.gguf llama-3.gguf
```

### Build

```bash
cargo build --release
```

### Run

```bash
./run.sh
```

### Result

```
Llama 3 Chat - Type 'quit' to exit
-----------------------------------
You: Hello
Assistant: Hello! How can I help you today?

You: quit
Goodbye!
```

### Build

```bash
cargo build
```

### Run

```bash
cargo run
```

### Result

```
❯ ./run.sh
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.08s
     Running `target/debug/swarms-rs-rust-agent-fun`
Result: User(User): Timestamp(millis): 1771207554237
Calculate 2+2, then 10*5, and explain your reasoning
RustAgent(Assistant): Timestamp(millis): 1771207556189
[Tool name]: calculator
[Tool args]: {"expression":"2+2"}
[Tool result]: "4"

[Tool name]: calculator
[Tool args]: {"expression":"10*5"}
[Tool result]: "50"


RustAgent(Assistant): Timestamp(millis): 1771207557263
[Tool name]: task_evaluator
[Tool args]: {"status":"Complete"}
[Tool result]: "Complete"
```
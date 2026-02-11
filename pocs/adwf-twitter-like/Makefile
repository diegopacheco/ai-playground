.PHONY: build run test fmt clippy clean db-up db-down help

build:
	cargo build --release

run:
	cargo run --release

test:
	cargo test

fmt:
	cargo fmt

clippy:
	cargo clippy

clean:
	cargo clean

db-up:
	podman-compose up -d

db-down:
	podman-compose down

help:
	@echo "Available targets:"
	@echo "  build    - Build the project in release mode"
	@echo "  run      - Run the server"
	@echo "  test     - Run tests"
	@echo "  fmt      - Format code"
	@echo "  clippy   - Run linter"
	@echo "  clean    - Clean build artifacts"
	@echo "  db-up    - Start PostgreSQL with podman-compose"
	@echo "  db-down  - Stop PostgreSQL"

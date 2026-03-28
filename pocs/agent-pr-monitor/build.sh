#!/bin/bash
cd frontend && bun install && bun run build && cd .. && cargo build --release

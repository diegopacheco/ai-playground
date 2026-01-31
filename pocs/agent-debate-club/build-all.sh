#!/bin/bash
cd backend && cargo build && cd ../frontend && bun run build

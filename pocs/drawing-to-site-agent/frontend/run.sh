#!/bin/bash
cd "$(dirname "$0")"
bun install && bun run dev

#!/bin/bash

export OPENAI_API_KEY=${OPENAI_API_KEY:?"OPENAI_API_KEY env var is required"}
cargo run

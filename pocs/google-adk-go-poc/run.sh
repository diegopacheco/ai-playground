#!/bin/bash

export OPENAI_API_KEY=${OPENAI_API_KEY}

if [ ! -f "go.mod" ]; then
  go mod init google-adk-go-poc
  go mod tidy
fi

go run main.go

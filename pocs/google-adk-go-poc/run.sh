#!/bin/bash

export OPENAI_API_KEY=${OPENAI_API_KEY}

go mod init google-adk-go-poc
go mod tidy
go run main.go

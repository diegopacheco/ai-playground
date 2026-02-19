#!/bin/bash

curl -s -X POST http://localhost:8081/agent/ask \
 -H "Content-Type: text/plain" \
 -d "How do virtual threads work in Java 25?" | glow

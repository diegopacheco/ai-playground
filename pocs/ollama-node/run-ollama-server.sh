#!/bin/bash

ollama pull deepseek-r1:1.5b
OLLAMA_USE_CPU=true ollama serve
#!/bin/bash
input=$(cat)
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

if [[ -n "$file_path" && "$file_path" == *.js ]]; then
  npx eslint "$file_path"
fi

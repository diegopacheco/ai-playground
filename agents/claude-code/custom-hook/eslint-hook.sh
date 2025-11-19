#!/bin/bash
input=$(cat)
echo "Hook triggered at $(date)" >> /tmp/eslint-hook.log
echo "Input: $input" >> /tmp/eslint-hook.log

file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')
echo "Extracted file_path: $file_path" >> /tmp/eslint-hook.log

if [[ -n "$file_path" && "$file_path" == *.js ]]; then
  echo "Running eslint on: $file_path" >> /tmp/eslint-hook.log
  npx eslint "$file_path" 2>&1 | tee -a /tmp/eslint-hook.log >&2
else
  echo "Skipped - file_path empty or not .js" >> /tmp/eslint-hook.log
fi

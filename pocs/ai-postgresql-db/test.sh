#!/bin/bash
set -e

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-..."
  exit 1
fi

echo "Running the LLM as a SQL primitive: classifying every review row inside the query engine."
echo

podman exec ai-postgres-db psql -U postgres -d aidb -c \
  "SELECT id, review, llm_classify(review) AS sentiment FROM reviews ORDER BY id;"

echo
echo "Filtering with the LLM in the WHERE-equivalent: only the negative reviews."
echo

podman exec ai-postgres-db psql -U postgres -d aidb -c \
  "SELECT id, review FROM (SELECT id, review, llm_classify(review) AS sentiment FROM reviews) t WHERE sentiment = 'NEGATIVE' ORDER BY id;"

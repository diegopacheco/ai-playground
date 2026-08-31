#!/bin/bash
set -e

cd "$(dirname "$0")"

PROMPTFOO=./node_modules/.bin/promptfoo

run_eval() {
  set +e
  $PROMPTFOO eval -c "$1" --no-cache --no-progress-bar
  local code=$?
  set -e
  if [ $code -ne 0 ] && [ $code -ne 100 ]; then
    echo "promptfoo failed on $1 with exit $code"
    exit $code
  fi
}

run_eval evals/classify.yaml
run_eval evals/extract.yaml

echo "opening the report on http://localhost:15500"
$PROMPTFOO view -p 15500

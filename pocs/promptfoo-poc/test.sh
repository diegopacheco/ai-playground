#!/bin/bash
set -e

cd "$(dirname "$0")"

RESULTS=.results
PROMPTFOO=./node_modules/.bin/promptfoo
mkdir -p "$RESULTS"

run_eval() {
  local config=$1
  local out=$2
  set +e
  $PROMPTFOO eval -c "$config" --no-cache --no-progress-bar -o "$out"
  local code=$?
  set -e
  if [ $code -ne 0 ] && [ $code -ne 100 ]; then
    echo "promptfoo failed on $config with exit $code"
    exit $code
  fi
}

echo "unit tests: prompt loader"
node --test src/prompt-loader.test.js

echo "eval: classify"
run_eval evals/classify.yaml "$RESULTS/classify.json"
node evals/gate.js "$RESULTS/classify.json" classify

echo "eval: extract"
run_eval evals/extract.yaml "$RESULTS/extract.json"
node evals/gate.js "$RESULTS/extract.json" extract

echo "test ok"

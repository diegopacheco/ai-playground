#!/bin/bash
set -e
cd "$(dirname "$0")"

./start.sh

BASE=http://localhost:8000

ask() {
  echo
  echo "Q: $1"
  curl -s -X POST $BASE/query \
    -H "Content-Type: application/json" \
    -d "{\"question\":\"$1\"}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('A:', d['answer']); print('Sources:', ', '.join('%s (%.3f)' % (s['file'], s['score']) for s in d['sources']))"
}

ask "How does the on-call rotation work?"
ask "What is the change freeze policy?"
ask "How much parental leave do primary caregivers get?"
ask "How do I get production database access?"

echo
echo "Open http://localhost:8000 in a browser for the UI. Run ./stop.sh when done."

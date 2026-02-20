#!/bin/bash
echo "Starting trip planning session..."
ID=$(curl -s -X POST http://localhost:8080/plan/start \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to plan a trip. Please help me plan the perfect vacation."}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

echo "Session ID: $ID"
echo ""

while true; do
  RESPONSE=$(curl -s "http://localhost:8080/plan/question/$ID")
  STATUS=$(echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['status'])")

  if [ "$STATUS" = "done" ]; then
    echo ""
    echo "=================================="
    echo "          TRIP PLAN               "
    echo "=================================="
    echo "$RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin)['result'])"
    break
  fi

  ANSWERS=$(echo "$RESPONSE" | python3 -c "
import json, sys

tty = open('/dev/tty', 'r')
data = json.load(sys.stdin)
answers = {}
for q in data.get('questions', []):
    print(f'Question: {q[\"question\"]}', file=sys.stderr)
    opts = q.get('options', [])
    for i, opt in enumerate(opts, 1):
        print(f'  {i}. {opt[\"label\"]}: {opt[\"description\"]}', file=sys.stderr)
    print('Your answer (enter number or type freely): ', end='', flush=True, file=sys.stderr)
    user_input = tty.readline().strip()
    try:
        idx = int(user_input) - 1
        answer = opts[idx]['label'] if 0 <= idx < len(opts) else user_input
    except (ValueError, IndexError):
        answer = user_input if user_input else (opts[0]['label'] if opts else 'Yes')
    print(f'=> {answer}', file=sys.stderr)
    print(file=sys.stderr)
    answers[q['question']] = answer
print(json.dumps({'answers': answers}))
")

  curl -s -X POST "http://localhost:8080/plan/answer/$ID" \
    -H "Content-Type: application/json" \
    -d "$ANSWERS" > /dev/null
done

#!/bin/bash

RESULTS_DIR="k6-results"

if [ ! -d "$RESULTS_DIR" ]; then
  echo "No results directory found. Run tests first."
  exit 1
fi

SUMMARY_FILES=$(find "$RESULTS_DIR" -name "*-summary.json" | sort)

if [ -z "$SUMMARY_FILES" ]; then
  echo "No summary files found in $RESULTS_DIR"
  exit 1
fi

echo "k6 Performance Test Results Analysis"
echo "====================================="
echo ""

for summary_file in $SUMMARY_FILES; do
  test_name=$(basename "$summary_file" "-summary.json")
  echo "Test: $test_name"
  echo "-----------------------------------"

  if [ -f "$summary_file" ]; then
    http_req_duration_p95=$(jq -r '.metrics.http_req_duration.values.["p(95)"] // "N/A"' "$summary_file")
    http_req_duration_p99=$(jq -r '.metrics.http_req_duration.values.["p(99)"] // "N/A"' "$summary_file")
    http_req_duration_avg=$(jq -r '.metrics.http_req_duration.values.avg // "N/A"' "$summary_file")
    http_req_failed=$(jq -r '.metrics.http_req_failed.values.rate // "N/A"' "$summary_file")
    http_reqs=$(jq -r '.metrics.http_reqs.values.count // "N/A"' "$summary_file")
    iterations=$(jq -r '.metrics.iterations.values.count // "N/A"' "$summary_file")
    vus_max=$(jq -r '.metrics.vus_max.values.max // "N/A"' "$summary_file")

    echo "  Response Times:"
    echo "    Average: ${http_req_duration_avg}ms"
    echo "    p95: ${http_req_duration_p95}ms"
    echo "    p99: ${http_req_duration_p99}ms"
    echo ""
    echo "  Request Stats:"
    echo "    Total Requests: $http_reqs"
    echo "    Failed Rate: $http_req_failed"
    echo "    Iterations: $iterations"
    echo "    Max VUs: $vus_max"
    echo ""

    error_rate=$(jq -r '.metrics.errors.values.rate // "N/A"' "$summary_file")
    if [ "$error_rate" != "N/A" ]; then
      echo "  Error Rate: $error_rate"
      echo ""
    fi

    tweets_created=$(jq -r '.metrics.tweets_created.values.count // "N/A"' "$summary_file")
    if [ "$tweets_created" != "N/A" ]; then
      echo "  Custom Metrics:"
      echo "    Tweets Created: $tweets_created"

      likes=$(jq -r '.metrics.likes_given.values.count // "N/A"' "$summary_file")
      [ "$likes" != "N/A" ] && echo "    Likes Given: $likes"

      retweets=$(jq -r '.metrics.retweets_given.values.count // "N/A"' "$summary_file")
      [ "$retweets" != "N/A" ] && echo "    Retweets Given: $retweets"

      comments=$(jq -r '.metrics.comments_created.values.count // "N/A"' "$summary_file")
      [ "$comments" != "N/A" ] && echo "    Comments Created: $comments"

      follows=$(jq -r '.metrics.follows_count.values.count // "N/A"' "$summary_file")
      [ "$follows" != "N/A" ] && echo "    Follows: $follows"

      feed_loads=$(jq -r '.metrics.feed_loads.values.count // "N/A"' "$summary_file")
      [ "$feed_loads" != "N/A" ] && echo "    Feed Loads: $feed_loads"

      echo ""
    fi

    thresholds=$(jq -r '.root_group.checks // [] | length' "$summary_file")
    if [ "$thresholds" != "0" ]; then
      passed=$(jq -r '[.root_group.checks[] | select(.passes == .fails + .passes)] | length' "$summary_file")
      echo "  Checks: $passed/$thresholds passed"
      echo ""
    fi
  else
    echo "  Summary file not found"
    echo ""
  fi
done

echo ""
echo "Detailed Results Location: $RESULTS_DIR/"
echo ""

LATEST_STRESS=$(find "$RESULTS_DIR" -name "stress-summary.json" | sort | tail -1)
if [ -f "$LATEST_STRESS" ]; then
  echo "Performance Summary (Stress Test)"
  echo "=================================="
  p95=$(jq -r '.metrics.http_req_duration.values.["p(95)"]' "$LATEST_STRESS")
  p99=$(jq -r '.metrics.http_req_duration.values.["p(99)"]' "$LATEST_STRESS")
  failed=$(jq -r '.metrics.http_req_failed.values.rate' "$LATEST_STRESS")

  echo "Response Time p95: ${p95}ms (Target: <500ms)"
  echo "Response Time p99: ${p99}ms (Target: <1000ms)"
  echo "Error Rate: $failed (Target: <0.01)"
  echo ""

  if (( $(echo "$p95 < 500" | bc -l) )) && (( $(echo "$p99 < 1000" | bc -l) )) && (( $(echo "$failed < 0.01" | bc -l) )); then
    echo "Status: ALL PERFORMANCE GOALS MET"
  else
    echo "Status: SOME PERFORMANCE GOALS NOT MET"
  fi
  echo ""
fi

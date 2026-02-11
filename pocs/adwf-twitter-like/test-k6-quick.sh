#!/bin/bash

echo "k6 Quick Test Demonstration"
echo "============================"
echo ""

echo "Step 1: Verifying setup..."
./verify-k6-setup.sh
if [ $? -ne 0 ]; then
  echo "Setup verification failed. Please check the output above."
  exit 1
fi

echo ""
echo "Step 2: Running baseline test (30 seconds)..."
echo "This will test the API with 10 concurrent users."
echo ""

./run-single-k6-test.sh baseline

echo ""
echo "Step 3: Analyzing results..."
echo ""

./analyze-k6-results.sh

echo ""
echo "Quick Test Complete!"
echo ""
echo "Next steps:"
echo "  - Run more tests: ./run-single-k6-test.sh <test-name>"
echo "  - Run full suite: ./run-k6-tests.sh"
echo "  - Generate report: ./generate-performance-report.sh"
echo ""
echo "Available tests: baseline, auth, user-profile, tweet-feed, social-interactions, load, stress"
echo ""

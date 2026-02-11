# k6 Performance Testing Files

This document lists all files created for k6 performance and stress testing.

## Test Scripts (k6-tests/)

1. **baseline-test.js**
   - Path: /private/tmp/test/k6-tests/baseline-test.js
   - Duration: 30 seconds
   - VUs: 10
   - Purpose: Establish baseline performance metrics
   - Tests: Feed retrieval, tweet creation, profile viewing

2. **auth-test.js**
   - Path: /private/tmp/test/k6-tests/auth-test.js
   - Duration: 2 minutes
   - VUs: Ramp 0→20→50→0
   - Purpose: Authentication endpoint stress testing
   - Tests: Registration, login, logout with performance tracking

3. **user-profile-test.js**
   - Path: /private/tmp/test/k6-tests/user-profile-test.js
   - Duration: 2.5 minutes
   - VUs: Ramp 0→15→40→15→0
   - Purpose: User profile operations testing
   - Tests: Profile view, update, followers/following lists

4. **tweet-feed-test.js**
   - Path: /private/tmp/test/k6-tests/tweet-feed-test.js
   - Duration: 3 minutes
   - VUs: Ramp 0→25→50→75→0
   - Purpose: Feed retrieval performance testing
   - Tests: Feed pagination, tweet creation, user tweets

5. **social-interactions-test.js**
   - Path: /private/tmp/test/k6-tests/social-interactions-test.js
   - Duration: 4.5 minutes
   - VUs: Ramp 0→30→60→30→0
   - Purpose: Social features stress testing
   - Tests: Likes/unlikes, retweets, comments, follow/unfollow

6. **load-test.js**
   - Path: /private/tmp/test/k6-tests/load-test.js
   - Duration: 4.5 minutes
   - VUs: Ramp 0→20→50→20→0
   - Purpose: Sustained load testing with realistic patterns
   - Tests: 40% reads, 60% writes mixed workload

7. **stress-test.js**
   - Path: /private/tmp/test/k6-tests/stress-test.js
   - Duration: 9 minutes
   - VUs: Ramp 0→25→50→100→50→0
   - Purpose: High concurrency stress testing
   - Tests: All operations under maximum load

## Execution Scripts

1. **run-k6-tests.sh**
   - Path: /private/tmp/test/run-k6-tests.sh
   - Purpose: Run all k6 tests sequentially
   - Features: Health check, result saving, progress reporting
   - Duration: ~30 minutes for full suite

2. **run-single-k6-test.sh**
   - Path: /private/tmp/test/run-single-k6-test.sh
   - Purpose: Run individual test by name
   - Usage: `./run-single-k6-test.sh <test-name>`
   - Features: Timestamp results, JSON export, summary generation

3. **verify-k6-setup.sh**
   - Path: /private/tmp/test/verify-k6-setup.sh
   - Purpose: Verify k6 and backend setup
   - Checks: k6 installation, server status, test files, API connectivity

4. **analyze-k6-results.sh**
   - Path: /private/tmp/test/analyze-k6-results.sh
   - Purpose: Analyze test results and display metrics
   - Features: Parse JSON summaries, display key metrics, goal validation

5. **generate-performance-report.sh**
   - Path: /private/tmp/test/generate-performance-report.sh
   - Purpose: Generate comprehensive performance report
   - Output: PERFORMANCE_REPORT.md with full analysis

## Documentation

1. **K6_PERFORMANCE_TESTS.md**
   - Path: /private/tmp/test/K6_PERFORMANCE_TESTS.md
   - Content: Complete testing documentation
   - Sections: Test overview, thresholds, metrics, optimization tips

2. **K6_QUICKSTART.md**
   - Path: /private/tmp/test/K6_QUICKSTART.md
   - Content: Quick start guide for k6 testing
   - Sections: Setup, running tests, analyzing results, troubleshooting

3. **k6-tests/README.md**
   - Path: /private/tmp/test/k6-tests/README.md
   - Content: Brief overview of test files
   - Purpose: Quick reference for test directory

4. **K6_FILES_SUMMARY.md**
   - Path: /private/tmp/test/K6_FILES_SUMMARY.md
   - Content: This file - complete file inventory

## Results Directory

**k6-results/**
- Created automatically when tests run
- Contains JSON results and summaries
- Files: `{test-name}-{timestamp}.json`, `{test-name}-summary.json`

## File Count Summary

- Test Scripts: 7 files
- Execution Scripts: 5 files
- Documentation: 4 files
- Total: 16 files created

## Quick Access

### Run Tests
```bash
./run-k6-tests.sh
./run-single-k6-test.sh baseline
```

### View Results
```bash
./analyze-k6-results.sh
./generate-performance-report.sh
```

### Documentation
```bash
cat K6_QUICKSTART.md
cat K6_PERFORMANCE_TESTS.md
cat k6-tests/README.md
```

## File Sizes

- Test scripts: ~2-6 KB each
- Shell scripts: ~1-5 KB each
- Documentation: ~5-10 KB each
- Total: ~60 KB

All files follow the user's guidelines with no comments and simple, well-written code.

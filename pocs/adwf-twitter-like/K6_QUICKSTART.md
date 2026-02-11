# k6 Performance Testing - Quick Start Guide

## Overview

This project includes comprehensive k6 performance and stress tests for the Twitter Clone API. The test suite covers all major endpoints and usage patterns.

## Installation

k6 is already installed. Verify with:
```bash
k6 version
```

If not installed:
```bash
brew install k6
```

## Setup Verification

Before running tests, verify the setup:
```bash
./verify-k6-setup.sh
```

This checks:
- k6 installation
- Backend server status
- Test files presence
- API connectivity

## Quick Test Run

Run a quick 30-second baseline test:
```bash
./run-single-k6-test.sh baseline
```

## Available Tests

| Test Name | Duration | Max VUs | Purpose |
|-----------|----------|---------|---------|
| baseline | 30s | 10 | Baseline performance metrics |
| auth | 2m | 50 | Authentication endpoints |
| user-profile | 2.5m | 40 | User profile operations |
| tweet-feed | 3m | 75 | Feed retrieval performance |
| social-interactions | 4.5m | 60 | Likes, retweets, comments |
| load | 4.5m | 50 | Sustained load testing |
| stress | 9m | 100 | High concurrency stress test |

## Running Individual Tests

```bash
./run-single-k6-test.sh <test-name>
```

Examples:
```bash
./run-single-k6-test.sh baseline
./run-single-k6-test.sh auth
./run-single-k6-test.sh load
./run-single-k6-test.sh stress
```

## Running All Tests

Run the complete test suite:
```bash
./run-k6-tests.sh
```

This takes approximately 30 minutes and runs all tests sequentially.

## Analyzing Results

View test results summary:
```bash
./analyze-k6-results.sh
```

Generate detailed performance report:
```bash
./generate-performance-report.sh
```

This creates `PERFORMANCE_REPORT.md` with complete analysis.

## Results Location

All test results are saved to:
```
k6-results/
  ├── baseline-{timestamp}.json
  ├── baseline-summary.json
  ├── auth-{timestamp}.json
  ├── auth-summary.json
  └── ...
```

## Performance Targets

- **p50 response time:** < 200ms
- **p95 response time:** < 500ms
- **p99 response time:** < 1000ms
- **Error rate:** < 1%
- **Success rate:** > 99%

## Understanding Test Scenarios

### Baseline Test
- 10 concurrent users
- 30 seconds duration
- Mixed operations: feed reads, tweet creation, profile views
- Purpose: Establish baseline metrics

### Load Test
- Ramps from 20 to 50 VUs
- 4.5 minutes duration
- Realistic mixed workload
- Purpose: Verify sustained performance

### Stress Test
- Ramps up to 100 VUs
- 9 minutes duration
- All operations under high load
- Purpose: Find system limits and breaking points

## Key Metrics

### Response Time Metrics
- `http_req_duration` - Total request duration
- `http_req_waiting` - Time to first byte
- `http_req_connecting` - Connection establishment time
- `http_req_sending` - Request sending time
- `http_req_receiving` - Response receiving time

### Request Metrics
- `http_reqs` - Total requests per second
- `http_req_failed` - Failed request rate
- `iterations` - Completed test iterations
- `vus` - Virtual users

### Custom Application Metrics
- `tweets_created` - Tweets created during test
- `likes_given` - Likes given
- `retweets_given` - Retweets made
- `comments_created` - Comments posted
- `follows_count` - Follow actions
- `feed_loads` - Feed retrieval count

## Troubleshooting

### Backend Not Running
```bash
./start.sh
curl http://localhost:8000/health
```

### High Error Rates
Check application logs:
```bash
tail -f logs/application.log
```

Check database:
```bash
./query-db.sh "SELECT count(*) FROM pg_stat_activity;"
```

### Slow Performance
Monitor resources during test:
```bash
docker stats
```

### Connection Issues
Verify BASE_URL:
```bash
export BASE_URL=http://localhost:8000
./run-single-k6-test.sh baseline
```

## Test Data Cleanup

Tests create test users. Clean up with:
```bash
./query-db.sh "
DELETE FROM comments WHERE user_id IN (SELECT id FROM users WHERE username LIKE '%_user_%' OR username LIKE 'auth_test_%');
DELETE FROM retweets WHERE user_id IN (SELECT id FROM users WHERE username LIKE '%_user_%' OR username LIKE 'auth_test_%');
DELETE FROM likes WHERE user_id IN (SELECT id FROM users WHERE username LIKE '%_user_%' OR username LIKE 'auth_test_%');
DELETE FROM tweets WHERE user_id IN (SELECT id FROM users WHERE username LIKE '%_user_%' OR username LIKE 'auth_test_%');
DELETE FROM followers WHERE follower_id IN (SELECT id FROM users WHERE username LIKE '%_user_%' OR username LIKE 'auth_test_%');
DELETE FROM users WHERE username LIKE '%_user_%' OR username LIKE 'auth_test_%';
"
```

## Workflow

1. Start backend server
2. Verify setup
3. Run baseline test
4. Analyze baseline results
5. Run targeted tests (auth, feed, etc.)
6. Run load test
7. Run stress test
8. Generate performance report
9. Identify bottlenecks
10. Optimize and re-test

## Next Steps

1. Run `./verify-k6-setup.sh` to ensure everything is ready
2. Run `./run-single-k6-test.sh baseline` for a quick test
3. Run `./analyze-k6-results.sh` to view results
4. Run `./run-k6-tests.sh` for complete suite
5. Run `./generate-performance-report.sh` for detailed report

## Documentation

- **K6_PERFORMANCE_TESTS.md** - Complete testing documentation
- **K6_QUICKSTART.md** - This file
- **k6-tests/README.md** - Test files overview
- **API_DOCUMENTATION.md** - API endpoints reference

## Support

For detailed information about each test scenario, thresholds, and performance optimization recommendations, see:
```bash
cat K6_PERFORMANCE_TESTS.md
```

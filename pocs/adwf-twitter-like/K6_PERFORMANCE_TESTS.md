# k6 Performance and Stress Testing

This directory contains comprehensive k6 performance and stress tests for the Twitter Clone API.

## Test Suite Overview

### 1. Baseline Test (`baseline-test.js`)
- Duration: 30 seconds
- Virtual Users: 10
- Purpose: Establish baseline performance metrics
- Operations: Feed retrieval, tweet creation, profile viewing
- Thresholds: p95 < 500ms, error rate < 1%

### 2. Authentication Test (`auth-test.js`)
- Duration: 2 minutes
- Virtual Users: Ramp 0 → 20 → 50 → 0
- Purpose: Test authentication endpoint performance
- Operations: Registration, login, logout
- Thresholds: Registration p95 < 600ms, Login p95 < 400ms

### 3. User Profile Test (`user-profile-test.js`)
- Duration: 2.5 minutes
- Virtual Users: Ramp 0 → 15 → 40 → 15 → 0
- Purpose: Test user profile operations
- Operations: Profile view, profile update, followers/following lists
- Thresholds: Profile view p95 < 400ms

### 4. Tweet Feed Test (`tweet-feed-test.js`)
- Duration: 3 minutes
- Virtual Users: Ramp 0 → 25 → 50 → 75 → 0
- Purpose: Test feed retrieval performance under load
- Operations: Feed retrieval with pagination, tweet creation, user tweets
- Thresholds: Feed p95 < 600ms, Tweet creation p95 < 500ms

### 5. Social Interactions Test (`social-interactions-test.js`)
- Duration: 4.5 minutes
- Virtual Users: Ramp 0 → 30 → 60 → 30 → 0
- Purpose: Test social interaction features
- Operations: Like/unlike, retweet/unretweet, comments, follow/unfollow
- Thresholds: p95 < 500ms, p99 < 1000ms

### 6. Load Test (`load-test.js`)
- Duration: 4.5 minutes
- Virtual Users: Ramp 0 → 20 → 50 → 20 → 0
- Purpose: Sustained load testing with realistic usage patterns
- Operations: Mixed workload with 40% reads, 60% writes
- Thresholds: p95 < 500ms, p99 < 1000ms

### 7. Stress Test (`stress-test.js`)
- Duration: 9 minutes
- Virtual Users: Ramp 0 → 25 → 50 → 100 → 50 → 0
- Purpose: Push system to limits and identify breaking points
- Operations: All API operations under high concurrency
- Thresholds: p95 < 500ms, p99 < 1000ms

## Performance Goals

### Response Time Targets
- p50 (median): < 200ms
- p95: < 500ms
- p99: < 1000ms

### Reliability Targets
- Error rate: < 1%
- HTTP failures: < 1%
- Success rate: > 99%

### Throughput Targets
- Baseline: 10 requests/second
- Load: 50 requests/second
- Stress: 100+ requests/second

## Running Tests

### Prerequisites
1. Install k6:
   ```bash
   brew install k6
   ```

2. Start the backend server:
   ```bash
   ./start.sh
   ```

3. Verify server is running:
   ```bash
   curl http://localhost:8000/health
   ```

### Run All Tests
```bash
./run-k6-tests.sh
```

### Run Individual Test
```bash
./run-single-k6-test.sh baseline
./run-single-k6-test.sh auth
./run-single-k6-test.sh user-profile
./run-single-k6-test.sh tweet-feed
./run-single-k6-test.sh social-interactions
./run-single-k6-test.sh load
./run-single-k6-test.sh stress
```

### Custom Base URL
```bash
BASE_URL=http://production-server:8000 ./run-k6-tests.sh
```

## Test Results

Results are saved to `k6-results/` directory:
- `{test-name}-{timestamp}.json` - Detailed test results
- `{test-name}-summary.json` - Test summary with metrics

### Key Metrics to Monitor

#### HTTP Metrics
- `http_req_duration` - Request duration (p50, p95, p99)
- `http_req_failed` - Failed request rate
- `http_reqs` - Total requests per second
- `http_req_blocked` - Time blocked before request
- `http_req_connecting` - Connection time
- `http_req_receiving` - Response receiving time
- `http_req_sending` - Request sending time
- `http_req_waiting` - Time to first byte

#### Custom Metrics
- `errors` - Application error rate
- `tweets_created` - Number of tweets created
- `likes_given` - Number of likes
- `retweets_given` - Number of retweets
- `comments_created` - Number of comments
- `follows_count` - Number of follows
- `feed_loads` - Number of feed loads
- `registration_duration` - Registration time
- `login_duration` - Login time
- `feed_duration` - Feed loading time
- `tweet_creation_duration` - Tweet creation time
- `profile_view_duration` - Profile view time

## Test Scenarios

### Baseline Test
Simulates normal usage with:
- 60% feed reads
- 25% tweet creation
- 15% profile viewing

### Load Test
Realistic mixed workload:
- 40% feed retrieval
- 20% tweet creation
- 15% likes
- 10% retweets
- 8% profile views
- 7% user tweet lists

### Stress Test
High concurrency with all operations:
- 35% feed retrieval
- 15% tweet creation
- 15% likes
- 10% retweets
- 10% comments
- 8% profile views
- 7% user tweet lists

## Interpreting Results

### Success Criteria
- All thresholds pass
- Error rate < 1%
- p95 response time < 500ms
- p99 response time < 1000ms
- No 5xx errors

### Warning Signs
- Increasing response times during ramp-up
- Error rate > 1%
- Connection timeouts
- Database connection pool exhaustion

### Common Bottlenecks
1. Database queries (N+1 problems)
2. Authentication/JWT validation
3. Connection pool limits
4. Memory allocation
5. CPU saturation

## Monitoring During Tests

Monitor system resources:
```bash
docker stats
```

Watch database connections:
```bash
./query-db.sh "SELECT count(*) FROM pg_stat_activity;"
```

Monitor logs:
```bash
docker logs -f twitter-clone-backend
```

## Optimization Recommendations

### Database
- Add indexes on frequently queried columns
- Optimize N+1 queries with joins
- Implement connection pooling
- Add query result caching
- Use read replicas for heavy read loads

### Application
- Implement response caching
- Use async operations where possible
- Optimize JWT validation
- Add rate limiting
- Implement request batching

### Infrastructure
- Scale horizontally with load balancer
- Use CDN for static assets
- Add Redis for session/cache
- Optimize database configuration
- Monitor and tune connection pools

## Test Data Cleanup

Tests create users and data. To clean up:
```bash
./query-db.sh "
DELETE FROM comments WHERE user_id IN (SELECT id FROM users WHERE username LIKE 'baseline_user_%' OR username LIKE 'load_user_%' OR username LIKE 'stress_user_%' OR username LIKE 'feed_user_%' OR username LIKE 'social_user_%' OR username LIKE 'profile_user_%' OR username LIKE 'auth_test_%');
DELETE FROM retweets WHERE user_id IN (SELECT id FROM users WHERE username LIKE 'baseline_user_%' OR username LIKE 'load_user_%' OR username LIKE 'stress_user_%' OR username LIKE 'feed_user_%' OR username LIKE 'social_user_%' OR username LIKE 'profile_user_%' OR username LIKE 'auth_test_%');
DELETE FROM likes WHERE user_id IN (SELECT id FROM users WHERE username LIKE 'baseline_user_%' OR username LIKE 'load_user_%' OR username LIKE 'stress_user_%' OR username LIKE 'feed_user_%' OR username LIKE 'social_user_%' OR username LIKE 'profile_user_%' OR username LIKE 'auth_test_%');
DELETE FROM tweets WHERE user_id IN (SELECT id FROM users WHERE username LIKE 'baseline_user_%' OR username LIKE 'load_user_%' OR username LIKE 'stress_user_%' OR username LIKE 'feed_user_%' OR username LIKE 'social_user_%' OR username LIKE 'profile_user_%' OR username LIKE 'auth_test_%');
DELETE FROM followers WHERE follower_id IN (SELECT id FROM users WHERE username LIKE 'baseline_user_%' OR username LIKE 'load_user_%' OR username LIKE 'stress_user_%' OR username LIKE 'feed_user_%' OR username LIKE 'social_user_%' OR username LIKE 'profile_user_%' OR username LIKE 'auth_test_%');
DELETE FROM users WHERE username LIKE 'baseline_user_%' OR username LIKE 'load_user_%' OR username LIKE 'stress_user_%' OR username LIKE 'feed_user_%' OR username LIKE 'social_user_%' OR username LIKE 'profile_user_%' OR username LIKE 'auth_test_%';
"
```

## CI/CD Integration

### GitHub Actions
```yaml
name: Performance Tests
on: [push]
jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Start services
        run: docker-compose up -d
      - name: Wait for services
        run: sleep 10
      - name: Run k6 tests
        run: ./run-k6-tests.sh
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: k6-results
          path: k6-results/
```

## Advanced Usage

### Running with k6 Cloud
```bash
k6 cloud k6-tests/stress-test.js
```

### Custom VU and Duration
```bash
k6 run --vus 50 --duration 5m k6-tests/load-test.js
```

### With HTTP Archive
```bash
k6 run --out json=results.json k6-tests/stress-test.js
```

### Real-time Metrics
```bash
k6 run --out influxdb=http://localhost:8086/k6 k6-tests/stress-test.js
```

## Troubleshooting

### Connection Refused
- Verify backend is running: `curl http://localhost:8000/health`
- Check BASE_URL environment variable

### High Error Rates
- Check database connection pool size
- Monitor database locks: `./query-db.sh "SELECT * FROM pg_locks;"`
- Check application logs

### Slow Response Times
- Profile database queries
- Check for N+1 query problems
- Monitor CPU and memory usage
- Check network latency

### Test Failures
- Ensure clean database state
- Verify thresholds are realistic
- Check for resource constraints
- Review test setup/teardown

## Next Steps

1. Run baseline test to establish metrics
2. Run individual feature tests
3. Run load test to verify sustained performance
4. Run stress test to find limits
5. Analyze bottlenecks
6. Optimize code/infrastructure
7. Re-run tests to verify improvements
8. Document performance benchmarks

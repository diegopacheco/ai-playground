# k6 Performance Tests

Comprehensive performance and stress testing suite for the Twitter Clone API.

## Quick Start

1. Verify setup:
```bash
../verify-k6-setup.sh
```

2. Run a quick test:
```bash
../run-single-k6-test.sh baseline
```

3. View results:
```bash
../analyze-k6-results.sh
```

## Test Files

- `baseline-test.js` - 30s baseline with 10 VUs
- `auth-test.js` - Authentication endpoints
- `user-profile-test.js` - User profile operations
- `tweet-feed-test.js` - Feed retrieval performance
- `social-interactions-test.js` - Likes, retweets, comments
- `load-test.js` - Load test up to 50 VUs
- `stress-test.js` - Stress test up to 100 VUs

## Performance Targets

- p95 response time: < 500ms
- p99 response time: < 1000ms
- Error rate: < 1%

## Full Documentation

See `/private/tmp/test/K6_PERFORMANCE_TESTS.md` for complete documentation.

# Integration Tests Execution Guide

## Overview

This guide provides step-by-step instructions for running the comprehensive integration tests for the Twitter clone application.

## Prerequisites

Before running the tests, ensure you have:

1. Rust and Cargo installed (version 1.93 or later)
2. Podman installed and running
3. Port 5433 available (for test database)
4. All dependencies installed (`cargo build`)

## Quick Start

To run all integration tests with automated setup and cleanup:

```bash
./run-integration-tests.sh
```

This script will:
1. Start a PostgreSQL test database on port 5433
2. Wait for database to be ready
3. Run all integration tests
4. Clean up the database after completion

## Manual Execution

If you prefer to run tests manually or need to debug:

### Step 1: Start Test Database

```bash
./setup-test-db.sh
```

Or manually:

```bash
podman run -d \
  --name twitter_test_db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=twitter_test \
  -p 5433:5432 \
  postgres:16-alpine
```

Wait for database to be ready:

```bash
while ! podman exec twitter_test_db pg_isready -U postgres > /dev/null 2>&1; do
  sleep 1
done
echo "Database is ready"
```

### Step 2: Set Environment Variables

```bash
export DATABASE_URL="postgres://postgres:postgres@localhost:5433/twitter_test"
export JWT_SECRET="test-secret-key-for-integration-tests"
```

### Step 3: Run Tests

Run all integration tests:

```bash
cargo test --test integration_tests -- --nocapture --test-threads=1
```

Run specific test:

```bash
cargo test --test integration_tests test_authentication_flow -- --nocapture
```

Run with verbose output:

```bash
RUST_LOG=debug cargo test --test integration_tests -- --nocapture --test-threads=1
```

### Step 4: Cleanup

```bash
podman stop twitter_test_db
podman rm twitter_test_db
```

## Test Categories

### Authentication Tests

Test authentication flow and validation:

```bash
cargo test --test integration_tests test_authentication -- --nocapture --test-threads=1
```

Includes:
- test_authentication_flow
- test_authentication_validation_errors

### User Operations Tests

Test user profile and social features:

```bash
cargo test --test integration_tests test_user -- --nocapture --test-threads=1
cargo test --test integration_tests test_follow -- --nocapture --test-threads=1
```

Includes:
- test_user_profile_operations
- test_follow_unfollow_operations

### Tweet Operations Tests

Test tweet CRUD operations:

```bash
cargo test --test integration_tests test_tweet -- --nocapture --test-threads=1
```

Includes:
- test_tweet_operations
- test_tweet_validation
- test_like_unlike_tweet
- test_retweet_operations

### Comment Tests

Test comment functionality:

```bash
cargo test --test integration_tests test_comment -- --nocapture --test-threads=1
```

### Feed Tests

Test feed generation and pagination:

```bash
cargo test --test integration_tests test_feed -- --nocapture --test-threads=1
```

### Security Tests

Test authorization and access control:

```bash
cargo test --test integration_tests test_unauthorized -- --nocapture --test-threads=1
```

### Data Persistence Tests

Test database persistence:

```bash
cargo test --test integration_tests test_data_persistence -- --nocapture --test-threads=1
cargo test --test integration_tests test_complex -- --nocapture --test-threads=1
```

## Understanding Test Output

### Successful Test

```
test test_authentication_flow ... ok
```

### Failed Test

```
test test_authentication_flow ... FAILED

failures:

---- test_authentication_flow stdout ----
thread 'test_authentication_flow' panicked at 'assertion failed: ...'
```

### Test with Output

Use `--nocapture` to see println! and debug output:

```
test test_authentication_flow ...
Setting up test database...
Running test...
Cleaning up...
ok
```

## Troubleshooting

### Database Connection Issues

**Problem**: Tests fail with connection errors

**Solutions**:
1. Check if database is running:
   ```bash
   podman ps | grep twitter_test_db
   ```

2. Check database logs:
   ```bash
   podman logs twitter_test_db
   ```

3. Verify port 5433 is available:
   ```bash
   lsof -i :5433
   ```

4. Restart database:
   ```bash
   podman stop twitter_test_db
   podman rm twitter_test_db
   ./setup-test-db.sh
   ```

### Migration Failures

**Problem**: Tests fail during database migration

**Solutions**:
1. Clean database and restart:
   ```bash
   podman exec twitter_test_db psql -U postgres -d twitter_test -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
   ```

2. Check migration files in `/private/tmp/test/migrations/`

3. Run migrations manually:
   ```bash
   export DATABASE_URL="postgres://postgres:postgres@localhost:5433/twitter_test"
   sqlx migrate run
   ```

### Test Failures

**Problem**: Specific tests fail

**Solutions**:
1. Run test with verbose output:
   ```bash
   RUST_LOG=twitter_clone=debug cargo test --test integration_tests test_name -- --nocapture
   ```

2. Check test isolation:
   ```bash
   cargo test --test integration_tests test_name -- --test-threads=1
   ```

3. Verify test data setup:
   - Check `setup_test_state()` function
   - Ensure database is clean before test

### Port Already in Use

**Problem**: Port 5433 is already in use

**Solutions**:
1. Find process using port:
   ```bash
   lsof -i :5433
   ```

2. Stop conflicting container:
   ```bash
   podman ps -a | grep 5433
   podman stop <container_id>
   ```

3. Use different port:
   Edit test DATABASE_URL to use different port

### Performance Issues

**Problem**: Tests run slowly

**Solutions**:
1. Use release mode:
   ```bash
   cargo test --test integration_tests --release
   ```

2. Increase connection pool:
   Edit `setup_test_state()` to increase max_connections

3. Run subset of tests:
   ```bash
   cargo test --test integration_tests test_authentication_flow
   ```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: twitter_test
        ports:
          - 5433:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v2

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run integration tests
        env:
          DATABASE_URL: postgres://postgres:postgres@localhost:5433/twitter_test
          JWT_SECRET: test-secret-key
        run: cargo test --test integration_tests -- --test-threads=1
```

### GitLab CI Example

```yaml
integration-tests:
  image: rust:latest
  services:
    - postgres:16-alpine
  variables:
    POSTGRES_DB: twitter_test
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
    DATABASE_URL: "postgres://postgres:postgres@postgres:5432/twitter_test"
    JWT_SECRET: "test-secret-key"
  script:
    - cargo test --test integration_tests -- --test-threads=1
```

## Test Coverage

Current test coverage includes:

- **Authentication**: 2 tests
- **User Operations**: 2 tests
- **Tweet Operations**: 2 tests
- **Social Interactions**: 5 tests
- **Security**: 1 test
- **Data Persistence**: 2 tests

**Total**: 13 comprehensive integration tests

Each test includes multiple assertions covering:
- HTTP status codes
- Response data validation
- Database state verification
- Error handling
- Authorization checks

## Expected Test Duration

- Individual test: 0.5-2 seconds
- Full test suite: 15-30 seconds (sequential execution)
- With database setup: 20-35 seconds

## Test Data

Tests use the following data patterns:

- **Usernames**: alice, bob, charlie, etc.
- **Emails**: username@test.com
- **Passwords**: password123 (6+ characters)
- **Tweets**: Various content strings (1-280 chars)
- **Comments**: Various comment strings (1-280 chars)

All test data is automatically cleaned up between tests.

## Success Criteria

All tests passing indicates:

- API endpoints work correctly
- Database operations are successful
- Authentication and authorization work
- Data persistence is reliable
- Error handling is appropriate
- Business logic is correct

## Next Steps

After running integration tests:

1. Review test output for any failures
2. Check code coverage (if using coverage tools)
3. Run performance tests
4. Deploy to staging environment
5. Run end-to-end tests with real frontend

## Additional Resources

- [INTEGRATION_TESTS.md](INTEGRATION_TESTS.md) - Detailed test documentation
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - API endpoint reference
- [DATABASE.md](DATABASE.md) - Database schema documentation
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup guide

## Support

For issues or questions:

1. Check troubleshooting section above
2. Review test output with `--nocapture`
3. Check database logs
4. Verify environment variables
5. Ensure all prerequisites are met

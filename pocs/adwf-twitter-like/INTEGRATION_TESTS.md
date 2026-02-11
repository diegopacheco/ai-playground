# Integration Tests Documentation

This document describes the comprehensive integration tests for the Twitter clone application.

## Overview

The integration tests verify the complete functionality of the API endpoints with real database interactions. Tests use a dedicated PostgreSQL database running in podman to ensure data persistence and realistic scenarios.

## Test Prerequisites

1. Podman installed and running
2. Rust and Cargo installed
3. PostgreSQL 16 available via podman
4. Port 5433 available for test database

## Running the Tests

### Quick Start

```bash
./run-integration-tests.sh
```

This script will:
1. Start a test PostgreSQL database on port 5433
2. Run all integration tests
3. Clean up the database after tests complete

### Manual Setup

```bash
./setup-test-db.sh
export DATABASE_URL="postgres://postgres:postgres@localhost:5433/twitter_test"
export JWT_SECRET="test-secret-key-for-integration-tests"
cargo test --test integration_tests -- --nocapture --test-threads=1
```

## Test Structure

### Test Database Setup

Each test uses `setup_test_state()` which:
- Connects to the test database
- Drops and recreates the schema for isolation
- Runs all migrations
- Returns a fresh AppState for testing

### Helper Functions

- `body_to_json()`: Converts response body to JSON
- `register_user()`: Registers a new user and returns token
- `login_user()`: Logs in a user and returns token

## Test Coverage

### 1. Authentication Flow Tests

#### test_authentication_flow
Tests the complete authentication lifecycle:
- User registration with valid credentials
- JWT token generation
- User login with correct credentials
- Logout functionality
- Verifies user data persistence

#### test_authentication_validation_errors
Tests authentication validation:
- Invalid email format rejection
- Short password rejection
- Duplicate username prevention
- Proper error status codes (400 BAD_REQUEST)

### 2. User Profile Operations

#### test_user_profile_operations
Tests user profile management:
- Fetching user profile with stats
- Updating display name and bio
- Profile data persistence
- Follower/following/tweet counts initialization

### 3. Follow/Unfollow Operations

#### test_follow_unfollow_operations
Tests social graph functionality:
- User following another user
- Follower count updates
- Following count updates
- Fetching followers list
- Fetching following list
- Unfollowing user
- Count decrements after unfollow
- Proper relationship tracking

### 4. Tweet Operations

#### test_tweet_operations
Tests core tweet functionality:
- Creating tweets with valid content
- Fetching individual tweets
- Fetching user's tweets with pagination
- Deleting tweets
- Tweet not found after deletion
- Tweet metadata (likes, retweets, comments counts)

#### test_tweet_validation
Tests tweet input validation:
- Empty content rejection
- Content over 280 characters rejection
- Proper validation error responses

### 5. Like/Unlike Operations

#### test_like_unlike_tweet
Tests tweet liking functionality:
- Liking a tweet
- Like count increments
- is_liked flag for user
- Unliking a tweet
- Like count decrements
- is_liked flag updates

### 6. Retweet Operations

#### test_retweet_operations
Tests retweet functionality:
- Retweeting a tweet
- Retweet count increments
- is_retweeted flag for user
- Removing retweet
- Retweet count decrements
- is_retweeted flag updates

### 7. Comment Operations

#### test_comment_operations
Tests commenting functionality:
- Adding comments to tweets
- Comment author tracking
- Fetching all comments for a tweet
- Comment count on tweet
- Deleting comments
- Comment list updates after deletion

### 8. Feed Generation

#### test_feed_generation
Tests the feed algorithm:
- Multiple users creating tweets
- User follows affecting feed content
- Feed returns tweets from followed users
- Feed pagination with limit parameter
- Feed pagination with offset parameter
- Proper ordering (most recent first)
- Feed includes user's own tweets

### 9. Authorization Tests

#### test_unauthorized_access
Tests security and authorization:
- Accessing protected endpoints without token (401)
- Accessing with invalid token (401)
- Users cannot modify other users' profiles (403)
- Proper error responses

### 10. Data Persistence

#### test_data_persistence
Tests database persistence:
- Data survives across requests
- Direct database queries confirm data
- Multiple app instances share same data
- User data persists correctly
- Tweet data persists correctly

### 11. Complex Scenarios

#### test_complex_user_interaction_scenario
Tests realistic multi-user interactions:
- User1 follows User2 and User3
- User2 creates a tweet
- User1 and User3 like the tweet
- User3 retweets the tweet
- User1 comments on the tweet
- Final tweet has correct counts:
  - 2 likes
  - 1 retweet
  - 1 comment
- User-specific flags work correctly

## Test Isolation

Tests run with `--test-threads=1` to ensure:
- No database conflicts between tests
- Each test gets a clean database state
- Predictable test execution order

Each test uses `setup_test_state()` which drops and recreates the database schema, ensuring complete isolation.

## Error Handling Tests

All tests verify proper error handling:
- HTTP status codes match API documentation
- Validation errors return 400 BAD_REQUEST
- Authentication failures return 401 UNAUTHORIZED
- Authorization failures return 403 FORBIDDEN
- Missing resources return 404 NOT_FOUND

## Performance Considerations

- Connection pooling (max 5 connections)
- Tests use realistic timeouts
- Async operations handled properly
- Database queries optimized with indexes

## Test Data Cleanup

The test database is:
- Created before tests run
- Cleaned between individual tests
- Removed after all tests complete
- Never interferes with production database

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
- Self-contained database setup
- No external dependencies
- Clean setup and teardown
- Exit codes indicate success/failure

## Troubleshooting

### Database Connection Failures
```bash
podman ps
podman logs twitter_test_db
```

### Test Failures
Run with verbose output:
```bash
cargo test --test integration_tests -- --nocapture
```

### Port Conflicts
Ensure port 5433 is available:
```bash
lsof -i :5433
```

### Cleanup Stuck Containers
```bash
podman stop twitter_test_db
podman rm twitter_test_db
```

## Test Metrics

Total test count: 13 comprehensive integration tests

Coverage areas:
- Authentication: 2 tests
- User operations: 2 tests
- Tweet operations: 2 tests
- Social interactions: 5 tests
- Security: 1 test
- Data persistence: 1 test
- Complex scenarios: 1 test

Each test includes multiple assertions to verify complete functionality.

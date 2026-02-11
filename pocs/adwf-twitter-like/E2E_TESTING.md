# End-to-End Testing with Playwright

This document describes the Playwright e2e test suite for the Twitter Clone application.

## Setup

Playwright has been installed and configured with TypeScript support.

### Installation

The following have been installed:
- @playwright/test - Playwright test framework
- Chromium browser for running tests

### Configuration

Playwright is configured in `playwright.config.ts` with:
- Test directory: `./e2e`
- Base URL: http://localhost:5173
- Screenshots on failure
- HTML reporter
- Auto-start of development server

## Test Structure

### Page Object Models

Tests use Page Object Models for maintainability:

#### `/e2e/pages/LoginPage.ts`
- Handles registration and login flows
- Methods: `register()`, `login()`, `waitForNavigation()`

#### `/e2e/pages/HomePage.ts`
- Handles home page interactions
- Methods: `createTweet()`, `likeTweet()`, `retweetTweet()`, `openTweetDetail()`

#### `/e2e/pages/ProfilePage.ts`
- Handles profile page interactions
- Methods: `followUser()`, `switchToTweetsTab()`, `switchToFollowersTab()`, `switchToFollowingTab()`

#### `/e2e/pages/TweetDetailPage.ts`
- Handles tweet detail page interactions
- Methods: `addComment()`, `deleteComment()`, `getCommentCount()`

### Helper Functions

#### `/e2e/helpers/auth.ts`
- `createTestUser()` - Creates a new test user and logs in
- `loginAsUser()` - Logs in as an existing user

## Test Suites

### Authentication Tests (`/e2e/auth.spec.ts`)
- User registration with validation
- Login with valid/invalid credentials
- Logout functionality
- Form toggle between login/signup
- Password minimum length validation
- Duplicate email handling
- Protected route redirection

**Test Count: 8 tests**

### Tweet Tests (`/e2e/tweets.spec.ts`)
- Create new tweet
- Character count display
- Button state management
- 280 character limit enforcement
- Like/unlike tweets
- Retweet functionality
- Navigate to tweet detail
- Delete own tweets
- Loading states

**Test Count: 11 tests**

### Comment Tests (`/e2e/comments.spec.ts`)
- Add comments to tweets
- Character count for comments
- Button state management
- 280 character limit for comments
- Multiple comments on a tweet
- Delete own comments
- Clear textarea after posting
- Loading states
- Comment count updates
- Empty state handling

**Test Count: 10 tests**

### Profile Tests (`/e2e/profile.spec.ts`)
- Navigate to own profile
- Display user information
- Show user tweets
- Switch between tabs (Tweets/Followers/Following)
- Empty state messages
- Follow button visibility
- Navigate to profile from tweets
- Loading states
- Handle invalid profiles

**Test Count: 12 tests**

### Follow/Unfollow Tests (`/e2e/follow.spec.ts`)
- Follow another user
- Show follower in followers list
- Show followed user in following list
- See followed user tweets in feed
- Empty feed when not following anyone
- Navigate to profile from followers/following lists

**Test Count: 7 tests**

### Feed Tests (`/e2e/feed.spec.ts`)
- Display feed on home page
- Show tweet composer
- Refresh feed after creating tweet
- Display tweets in reverse chronological order
- Loading spinner
- Error message handling
- Display user avatars
- Display username and handle
- Display timestamps
- Display interaction buttons
- Display counts for likes, retweets, and comments

**Test Count: 12 tests**

### Navigation Tests (`/e2e/navigation.spec.ts`)
- Display navigation bar on all pages
- Navigate to home/profile from nav bar
- Display user handle in nav bar
- Navigate to login after logout
- Sticky navigation bar
- Navigate to tweet detail from feed
- Navigate to user profile from tweet
- Navigate back from tweet detail
- Twitter Clone branding
- Browser back/forward buttons

**Test Count: 13 tests**

### Responsive Design Tests (`/e2e/responsive.spec.ts`)
- Mobile viewport (375x667)
- Tablet viewport (768x1024)
- Desktop viewport (1920x1080)
- Responsive components on mobile
- Scrollable feed on mobile
- Responsive pages on mobile
- Text wrapping on mobile
- Touch-friendly buttons
- Orientation change handling

**Test Count: 13 tests**

## Total Test Coverage

**Total: 86 end-to-end tests**

## Running Tests

### Prerequisites

Ensure both backend and frontend servers are running:
```bash
cargo run --release
npm run dev
```

### Run All Tests
```bash
npm run test:e2e
```

### Run Tests in UI Mode
```bash
npm run test:e2e:ui
```

### Run Tests in Headed Mode
```bash
npm run test:e2e:headed
```

### View Test Report
```bash
npm run test:e2e:report
```

### Run Specific Test File
```bash
npx playwright test e2e/auth.spec.ts
```

### Run Tests with Specific Tag
```bash
npx playwright test --grep "should register"
```

## Automated Test Runner

Use the provided script to automatically start servers and run tests:
```bash
./run-e2e-tests.sh
```

This script will:
1. Start the backend server on port 8000
2. Wait for backend to be ready
3. Start the frontend server on port 5173
4. Wait for frontend to be ready
5. Run all Playwright tests
6. Clean up processes on exit

## Test Best Practices

### Selectors
Tests use a combination of:
- Semantic HTML selectors (id, aria-label)
- Text content for better readability
- CSS classes sparingly
- Stable locators that won't break with styling changes

### Wait Strategies
Tests implement proper wait strategies:
- `waitForSelector()` for dynamic elements
- `waitForTimeout()` for API responses (500-1000ms)
- `waitForNavigation()` for page transitions
- Automatic waiting with Playwright's built-in mechanisms

### Independence
Each test:
- Creates its own test user
- Is isolated from other tests
- Cleans up after itself
- Can run in parallel

### Data Management
- Unique usernames/emails using timestamps
- Predictable test data
- No shared state between tests

## Debugging Tests

### Debug Mode
```bash
npx playwright test --debug
```

### Trace Viewer
Traces are captured on first retry. View them with:
```bash
npx playwright show-trace trace.zip
```

### Screenshots
Screenshots are automatically captured on failure in `test-results/`

### Verbose Output
```bash
npx playwright test --reporter=list
```

## CI/CD Integration

The test suite is ready for CI/CD:
- Retries configured (2 retries in CI)
- Parallel execution disabled in CI
- HTML reporter for test results
- Screenshots and traces for debugging failures

Example GitHub Actions:
```yaml
- name: Run E2E Tests
  run: npm run test:e2e
```

## Test Maintenance

### Adding New Tests
1. Create test file in `/e2e/` directory
2. Import necessary page objects
3. Use `createTestUser()` helper for authentication
4. Follow naming convention: `feature.spec.ts`

### Updating Page Objects
When UI changes:
1. Update relevant page object in `/e2e/pages/`
2. Run affected tests to verify
3. Update selectors if needed

### Test Data
Test users are created with pattern:
```
username: {testname}_{timestamp}
email: {testname}_{timestamp}@test.com
password: password123
```

## Known Limitations

1. Tests require both backend and frontend to be running
2. Backend must be on port 8000
3. Frontend must be on port 5173
4. Database should be in a clean state for reliable results
5. Some tests use timeouts for API responses (may need adjustment for slower systems)

## Future Improvements

1. Add visual regression testing
2. Add accessibility testing
3. Add performance testing
4. Mock API responses for faster tests
5. Add test data fixtures
6. Add custom reporters
7. Add test parallelization strategies
8. Add mobile device emulation tests

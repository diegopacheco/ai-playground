# Playwright E2E Tests Quick Reference

## Test Files

- `auth.spec.ts` - Authentication and authorization tests
- `tweets.spec.ts` - Tweet creation, like, retweet, delete tests
- `comments.spec.ts` - Comment creation and management tests
- `profile.spec.ts` - User profile page tests
- `follow.spec.ts` - Follow/unfollow functionality tests
- `feed.spec.ts` - Feed display and interaction tests
- `navigation.spec.ts` - Navigation and routing tests
- `responsive.spec.ts` - Responsive design tests

## Page Objects

Located in `/e2e/pages/`:
- `LoginPage.ts` - Login and registration page
- `HomePage.ts` - Home feed page
- `ProfilePage.ts` - User profile page
- `TweetDetailPage.ts` - Tweet detail page

## Helpers

Located in `/e2e/helpers/`:
- `auth.ts` - Authentication helper functions

## Quick Commands

```bash
npm run test:e2e
npm run test:e2e:ui
npm run test:e2e:headed
npm run test:e2e:report
```

## Running Individual Suites

```bash
npx playwright test e2e/auth.spec.ts
npx playwright test e2e/tweets.spec.ts
npx playwright test e2e/comments.spec.ts
npx playwright test e2e/profile.spec.ts
npx playwright test e2e/follow.spec.ts
npx playwright test e2e/feed.spec.ts
npx playwright test e2e/navigation.spec.ts
npx playwright test e2e/responsive.spec.ts
```

## Test Statistics

- Total Tests: 86
- Authentication: 8 tests
- Tweets: 11 tests
- Comments: 10 tests
- Profile: 12 tests
- Follow: 7 tests
- Feed: 12 tests
- Navigation: 13 tests
- Responsive: 13 tests

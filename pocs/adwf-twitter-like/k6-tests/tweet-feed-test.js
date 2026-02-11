import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Counter, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const feedRequestsTotal = new Counter('feed_requests_total');
const feedDuration = new Trend('feed_duration');
const tweetCreationDuration = new Trend('tweet_creation_duration');
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  stages: [
    { duration: '30s', target: 25 },
    { duration: '1m', target: 50 },
    { duration: '1m', target: 75 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.01'],
    errors: ['rate<0.01'],
    feed_duration: ['p(95)<600'],
    tweet_creation_duration: ['p(95)<500'],
  },
};

export function setup() {
  const registerUrl = `${BASE_URL}/api/auth/register`;
  const testUsers = [];

  for (let i = 0; i < 75; i++) {
    const userData = {
      username: `feed_user_${Date.now()}_${i}`,
      email: `feed_user_${Date.now()}_${i}@test.com`,
      password: 'password123'
    };

    const res = http.post(registerUrl, JSON.stringify(userData), {
      headers: { 'Content-Type': 'application/json' },
    });

    if (res.status === 201) {
      const body = JSON.parse(res.body);
      testUsers.push({
        token: body.token,
        userId: body.user.id,
        username: userData.username
      });
    }

    sleep(0.1);
  }

  for (let i = 0; i < Math.min(testUsers.length - 1, 50); i++) {
    const follower = testUsers[i];
    const followee = testUsers[(i + 1) % testUsers.length];

    const headers = {
      'Authorization': `Bearer ${follower.token}`
    };

    http.post(`${BASE_URL}/api/users/${followee.userId}/follow`, null, { headers });
    sleep(0.1);
  }

  for (let i = 0; i < Math.min(testUsers.length, 30); i++) {
    const user = testUsers[i];
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${user.token}`
    };

    for (let j = 0; j < 5; j++) {
      const tweetContent = {
        content: `Initial tweet ${j} from ${user.username} for feed population`
      };

      http.post(`${BASE_URL}/api/tweets`, JSON.stringify(tweetContent), { headers });
      sleep(0.05);
    }
  }

  return { users: testUsers };
}

export default function(data) {
  const user = data.users[__VU % data.users.length];
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${user.token}`
  };

  const scenario = Math.random();

  if (scenario < 0.6) {
    const limits = [10, 20, 50];
    const limit = limits[Math.floor(Math.random() * limits.length)];
    const offset = Math.floor(Math.random() * 3) * limit;

    const feedStart = Date.now();
    const feedRes = http.get(
      `${BASE_URL}/api/tweets/feed?limit=${limit}&offset=${offset}`,
      { headers }
    );
    feedDuration.add(Date.now() - feedStart);
    feedRequestsTotal.add(1);

    check(feedRes, {
      'feed status 200': (r) => r.status === 200,
      'feed is array': (r) => {
        try {
          const body = JSON.parse(r.body);
          return Array.isArray(body);
        } catch {
          return false;
        }
      },
      'feed response time acceptable': (r) => r.timings.duration < 600,
    }) || errorRate.add(1);
  } else if (scenario < 0.9) {
    const tweetContent = {
      content: `Feed test tweet at ${Date.now()} from ${user.username}`
    };

    const tweetStart = Date.now();
    const createRes = http.post(
      `${BASE_URL}/api/tweets`,
      JSON.stringify(tweetContent),
      { headers }
    );
    tweetCreationDuration.add(Date.now() - tweetStart);

    check(createRes, {
      'tweet created': (r) => r.status === 201,
      'tweet has content': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.content === tweetContent.content;
        } catch {
          return false;
        }
      }
    }) || errorRate.add(1);
  } else {
    const otherUser = data.users[Math.floor(Math.random() * data.users.length)];
    const limit = 20;
    const offset = 0;

    const userTweetsRes = http.get(
      `${BASE_URL}/api/tweets/user/${otherUser.userId}?limit=${limit}&offset=${offset}`,
      { headers }
    );

    check(userTweetsRes, {
      'user tweets status 200': (r) => r.status === 200,
      'user tweets is array': (r) => {
        try {
          const body = JSON.parse(r.body);
          return Array.isArray(body);
        } catch {
          return false;
        }
      }
    }) || errorRate.add(1);
  }

  sleep(Math.random() * 2 + 0.5);
}

export function teardown(data) {
  console.log(`Tweet feed stress test completed with ${data.users.length} users`);
}

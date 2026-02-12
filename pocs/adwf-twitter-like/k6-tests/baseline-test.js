import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  stages: [
    { duration: '30s', target: 10 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
    errors: ['rate<0.01'],
  },
};

const users = [];

export function setup() {
  const registerUrl = `${BASE_URL}/api/auth/register`;
  const loginUrl = `${BASE_URL}/api/auth/login`;
  const testUsers = [];
  const ts = Date.now();

  for (let i = 0; i < 10; i++) {
    const username = `bl_${ts}_${i}`;
    const userData = {
      username: username,
      email: `${username}@test.com`,
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
        username: username
      });
    } else {
      const loginRes = http.post(loginUrl, JSON.stringify({ username: username, password: 'password123' }), {
        headers: { 'Content-Type': 'application/json' },
      });
      if (loginRes.status === 200) {
        const body = JSON.parse(loginRes.body);
        testUsers.push({
          token: body.token,
          userId: body.user.id,
          username: username
        });
      }
    }

    sleep(0.1);
  }

  return { users: testUsers };
}

export default function(data) {
  if (!data.users || data.users.length === 0) {
    return;
  }
  const user = data.users[__VU % data.users.length];
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${user.token}`
  };

  const feedRes = http.get(`${BASE_URL}/api/tweets/feed?limit=20`, { headers });
  check(feedRes, {
    'feed status 200': (r) => r.status === 200,
  }) || errorRate.add(1);

  sleep(1);

  const tweetContent = {
    content: `Baseline test tweet at ${Date.now()}`
  };
  const createRes = http.post(`${BASE_URL}/api/tweets`, JSON.stringify(tweetContent), { headers });
  check(createRes, {
    'tweet created': (r) => r.status === 201,
  }) || errorRate.add(1);

  if (createRes.status === 201) {
    const tweet = JSON.parse(createRes.body);

    const getTweetRes = http.get(`${BASE_URL}/api/tweets/${tweet.id}`, { headers });
    check(getTweetRes, {
      'get tweet 200': (r) => r.status === 200,
    }) || errorRate.add(1);
  }

  sleep(1);

  const profileRes = http.get(`${BASE_URL}/api/users/${user.userId}`, { headers });
  check(profileRes, {
    'profile status 200': (r) => r.status === 200,
  }) || errorRate.add(1);

  sleep(1);
}

export function teardown(data) {
  console.log(`Baseline test completed with ${data.users.length} users`);
}

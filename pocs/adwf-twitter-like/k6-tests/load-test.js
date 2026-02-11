import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Counter } from 'k6/metrics';

const errorRate = new Rate('errors');
const tweetsCreated = new Counter('tweets_created');
const likesGiven = new Counter('likes_given');
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  stages: [
    { duration: '1m', target: 20 },
    { duration: '2m', target: 50 },
    { duration: '1m', target: 20 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.01'],
    errors: ['rate<0.01'],
  },
};

export function setup() {
  const registerUrl = `${BASE_URL}/api/auth/register`;
  const testUsers = [];

  for (let i = 0; i < 50; i++) {
    const userData = {
      username: `load_user_${Date.now()}_${i}`,
      email: `load_user_${Date.now()}_${i}@test.com`,
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

  const tweetIds = [];
  for (let i = 0; i < Math.min(10, testUsers.length); i++) {
    const user = testUsers[i];
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${user.token}`
    };

    const tweetContent = {
      content: `Setup tweet ${i} for load testing`
    };

    const res = http.post(`${BASE_URL}/api/tweets`, JSON.stringify(tweetContent), { headers });
    if (res.status === 201) {
      const tweet = JSON.parse(res.body);
      tweetIds.push(tweet.id);
    }

    sleep(0.1);
  }

  return { users: testUsers, tweetIds };
}

export default function(data) {
  const user = data.users[__VU % data.users.length];
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${user.token}`
  };

  const scenario = Math.random();

  if (scenario < 0.4) {
    const feedRes = http.get(`${BASE_URL}/api/tweets/feed?limit=20`, { headers });
    check(feedRes, {
      'feed loaded': (r) => r.status === 200,
    }) || errorRate.add(1);
  } else if (scenario < 0.6) {
    const tweetContent = {
      content: `Load test tweet from ${user.username} at ${Date.now()}`
    };
    const createRes = http.post(`${BASE_URL}/api/tweets`, JSON.stringify(tweetContent), { headers });
    check(createRes, {
      'tweet created': (r) => r.status === 201,
    }) || errorRate.add(1);

    if (createRes.status === 201) {
      tweetsCreated.add(1);
    }
  } else if (scenario < 0.75) {
    if (data.tweetIds.length > 0) {
      const tweetId = data.tweetIds[Math.floor(Math.random() * data.tweetIds.length)];
      const likeRes = http.post(`${BASE_URL}/api/tweets/${tweetId}/like`, null, { headers });
      check(likeRes, {
        'like succeeded': (r) => r.status === 201 || r.status === 200,
      }) || errorRate.add(1);

      if (likeRes.status === 201 || likeRes.status === 200) {
        likesGiven.add(1);
      }
    }
  } else if (scenario < 0.85) {
    if (data.tweetIds.length > 0) {
      const tweetId = data.tweetIds[Math.floor(Math.random() * data.tweetIds.length)];
      const retweetRes = http.post(`${BASE_URL}/api/tweets/${tweetId}/retweet`, null, { headers });
      check(retweetRes, {
        'retweet succeeded': (r) => r.status === 201 || r.status === 200,
      });
    }
  } else {
    const profileRes = http.get(`${BASE_URL}/api/users/${user.userId}`, { headers });
    check(profileRes, {
      'profile loaded': (r) => r.status === 200,
    }) || errorRate.add(1);
  }

  sleep(Math.random() * 2 + 0.5);
}

export function teardown(data) {
  console.log(`Load test completed`);
  console.log(`Users: ${data.users.length}`);
  console.log(`Test tweets: ${data.tweetIds.length}`);
}

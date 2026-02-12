import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Counter, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const tweetsCreated = new Counter('tweets_created');
const likesGiven = new Counter('likes_given');
const retweetsGiven = new Counter('retweets_given');
const commentsCreated = new Counter('comments_created');
const feedLoads = new Counter('feed_loads');
const authLatency = new Trend('auth_latency');
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  stages: [
    { duration: '1m', target: 25 },
    { duration: '2m', target: 50 },
    { duration: '3m', target: 100 },
    { duration: '2m', target: 50 },
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.01'],
    errors: ['rate<0.01'],
  },
};

export function setup() {
  const registerUrl = `${BASE_URL}/api/auth/register`;
  const loginUrl = `${BASE_URL}/api/auth/login`;
  const testUsers = [];
  const ts = Date.now();

  for (let i = 0; i < 100; i++) {
    const username = `st_${ts}_${i}`;
    const userData = {
      username: username,
      email: `${username}@test.com`,
      password: 'password123'
    };

    const startAuth = Date.now();
    const res = http.post(registerUrl, JSON.stringify(userData), {
      headers: { 'Content-Type': 'application/json' },
    });
    authLatency.add(Date.now() - startAuth);

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

  const tweetIds = [];
  for (let i = 0; i < Math.min(20, testUsers.length); i++) {
    const user = testUsers[i];
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${user.token}`
    };

    const tweetContent = {
      content: `Stress test initial tweet ${i} - ready for high load testing`
    };

    const res = http.post(`${BASE_URL}/api/tweets`, JSON.stringify(tweetContent), { headers });
    if (res.status === 201) {
      const tweet = JSON.parse(res.body);
      tweetIds.push(tweet.id);
    }

    sleep(0.1);
  }

  const followPairs = [];
  for (let i = 0; i < Math.min(testUsers.length - 1, 30); i++) {
    const follower = testUsers[i];
    const followee = testUsers[(i + 1) % testUsers.length];

    const headers = {
      'Authorization': `Bearer ${follower.token}`
    };

    http.post(`${BASE_URL}/api/users/${followee.userId}/follow`, null, { headers });
    followPairs.push({ followerId: follower.userId, followeeId: followee.userId });

    sleep(0.1);
  }

  return { users: testUsers, tweetIds, followPairs };
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

  const scenario = Math.random();

  if (scenario < 0.35) {
    const feedRes = http.get(`${BASE_URL}/api/tweets/feed?limit=20&offset=0`, { headers });
    check(feedRes, {
      'feed loaded': (r) => r.status === 200,
      'feed has data': (r) => {
        try {
          const body = JSON.parse(r.body);
          return Array.isArray(body);
        } catch {
          return false;
        }
      }
    }) || errorRate.add(1);

    if (feedRes.status === 200) {
      feedLoads.add(1);
    }
  } else if (scenario < 0.5) {
    const tweetContent = {
      content: `Stress test tweet from ${user.username} - load testing at ${Date.now()}`
    };
    const createRes = http.post(`${BASE_URL}/api/tweets`, JSON.stringify(tweetContent), { headers });
    check(createRes, {
      'tweet created': (r) => r.status === 201,
      'tweet has id': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.id > 0;
        } catch {
          return false;
        }
      }
    }) || errorRate.add(1);

    if (createRes.status === 201) {
      tweetsCreated.add(1);
    }
  } else if (scenario < 0.65) {
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
  } else if (scenario < 0.75) {
    if (data.tweetIds.length > 0) {
      const tweetId = data.tweetIds[Math.floor(Math.random() * data.tweetIds.length)];
      const retweetRes = http.post(`${BASE_URL}/api/tweets/${tweetId}/retweet`, null, { headers });
      check(retweetRes, {
        'retweet succeeded': (r) => r.status === 201 || r.status === 200,
      });

      if (retweetRes.status === 201 || retweetRes.status === 200) {
        retweetsGiven.add(1);
      }
    }
  } else if (scenario < 0.85) {
    if (data.tweetIds.length > 0) {
      const tweetId = data.tweetIds[Math.floor(Math.random() * data.tweetIds.length)];
      const commentContent = {
        content: `Comment from ${user.username} under stress test`
      };
      const commentRes = http.post(
        `${BASE_URL}/api/tweets/${tweetId}/comments`,
        JSON.stringify(commentContent),
        { headers }
      );
      check(commentRes, {
        'comment created': (r) => r.status === 201,
      }) || errorRate.add(1);

      if (commentRes.status === 201) {
        commentsCreated.add(1);
      }
    }
  } else if (scenario < 0.92) {
    const profileRes = http.get(`${BASE_URL}/api/users/${user.userId}`, { headers });
    check(profileRes, {
      'profile loaded': (r) => r.status === 200,
      'profile has username': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.username === user.username;
        } catch {
          return false;
        }
      }
    }) || errorRate.add(1);
  } else {
    const otherUser = data.users[Math.floor(Math.random() * data.users.length)];
    const tweetsRes = http.get(
      `${BASE_URL}/api/tweets/user/${otherUser.userId}?limit=10`,
      { headers }
    );
    check(tweetsRes, {
      'user tweets loaded': (r) => r.status === 200,
    }) || errorRate.add(1);
  }

  sleep(Math.random() * 1.5 + 0.2);
}

export function teardown(data) {
  console.log(`Stress test completed`);
  console.log(`Total users: ${data.users.length}`);
  console.log(`Test tweets: ${data.tweetIds.length}`);
  console.log(`Follow pairs: ${data.followPairs.length}`);
}

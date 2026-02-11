import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Counter } from 'k6/metrics';

const errorRate = new Rate('errors');
const likesCount = new Counter('likes_count');
const retweetsCount = new Counter('retweets_count');
const commentsCount = new Counter('comments_count');
const followsCount = new Counter('follows_count');
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  stages: [
    { duration: '1m', target: 30 },
    { duration: '2m', target: 60 },
    { duration: '1m', target: 30 },
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

  for (let i = 0; i < 60; i++) {
    const userData = {
      username: `social_user_${Date.now()}_${i}`,
      email: `social_user_${Date.now()}_${i}@test.com`,
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
  for (let i = 0; i < Math.min(testUsers.length, 25); i++) {
    const user = testUsers[i];
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${user.token}`
    };

    const tweetContent = {
      content: `Social interaction test tweet ${i} from ${user.username}`
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

  const action = Math.random();

  if (action < 0.3 && data.tweetIds.length > 0) {
    const tweetId = data.tweetIds[Math.floor(Math.random() * data.tweetIds.length)];

    const likeRes = http.post(`${BASE_URL}/api/tweets/${tweetId}/like`, null, { headers });
    const likeSuccess = check(likeRes, {
      'like created or exists': (r) => r.status === 201 || r.status === 200,
    });

    if (likeSuccess) {
      likesCount.add(1);
    } else {
      errorRate.add(1);
    }

    sleep(0.5);

    const unlikeRes = http.del(`${BASE_URL}/api/tweets/${tweetId}/like`, null, { headers });
    check(unlikeRes, {
      'unlike succeeded': (r) => r.status === 204 || r.status === 200,
    }) || errorRate.add(1);

  } else if (action < 0.5 && data.tweetIds.length > 0) {
    const tweetId = data.tweetIds[Math.floor(Math.random() * data.tweetIds.length)];

    const retweetRes = http.post(`${BASE_URL}/api/tweets/${tweetId}/retweet`, null, { headers });
    const retweetSuccess = check(retweetRes, {
      'retweet created or exists': (r) => r.status === 201 || r.status === 200,
    });

    if (retweetSuccess) {
      retweetsCount.add(1);
    } else {
      errorRate.add(1);
    }

    sleep(0.5);

    const unretweetRes = http.del(`${BASE_URL}/api/tweets/${tweetId}/retweet`, null, { headers });
    check(unretweetRes, {
      'unretweet succeeded': (r) => r.status === 204 || r.status === 200,
    }) || errorRate.add(1);

  } else if (action < 0.7 && data.tweetIds.length > 0) {
    const tweetId = data.tweetIds[Math.floor(Math.random() * data.tweetIds.length)];
    const commentContent = {
      content: `Test comment from ${user.username} at ${Date.now()}`
    };

    const commentRes = http.post(
      `${BASE_URL}/api/tweets/${tweetId}/comments`,
      JSON.stringify(commentContent),
      { headers }
    );

    const commentSuccess = check(commentRes, {
      'comment created': (r) => r.status === 201,
      'comment has content': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.content === commentContent.content;
        } catch {
          return false;
        }
      }
    });

    if (commentSuccess) {
      commentsCount.add(1);
    } else {
      errorRate.add(1);
    }

    if (commentRes.status === 201) {
      const comment = JSON.parse(commentRes.body);

      sleep(0.5);

      const getCommentsRes = http.get(
        `${BASE_URL}/api/tweets/${tweetId}/comments`,
        { headers }
      );
      check(getCommentsRes, {
        'get comments succeeded': (r) => r.status === 200,
        'comments is array': (r) => {
          try {
            const body = JSON.parse(r.body);
            return Array.isArray(body);
          } catch {
            return false;
          }
        }
      }) || errorRate.add(1);
    }

  } else {
    const otherUser = data.users[Math.floor(Math.random() * data.users.length)];

    if (otherUser.userId !== user.userId) {
      const followRes = http.post(
        `${BASE_URL}/api/users/${otherUser.userId}/follow`,
        null,
        { headers }
      );

      const followSuccess = check(followRes, {
        'follow created or exists': (r) => r.status === 201 || r.status === 200,
      });

      if (followSuccess) {
        followsCount.add(1);
      } else {
        errorRate.add(1);
      }

      sleep(0.5);

      const unfollowRes = http.del(
        `${BASE_URL}/api/users/${otherUser.userId}/follow`,
        null,
        { headers }
      );
      check(unfollowRes, {
        'unfollow succeeded': (r) => r.status === 204 || r.status === 200,
      }) || errorRate.add(1);
    }
  }

  sleep(Math.random() * 2 + 0.5);
}

export function teardown(data) {
  console.log(`Social interactions test completed`);
  console.log(`Total users: ${data.users.length}`);
  console.log(`Test tweets: ${data.tweetIds.length}`);
}

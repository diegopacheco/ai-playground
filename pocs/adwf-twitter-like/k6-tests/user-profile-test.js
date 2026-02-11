import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Counter, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const profileUpdates = new Counter('profile_updates');
const profileViewDuration = new Trend('profile_view_duration');
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  stages: [
    { duration: '30s', target: 15 },
    { duration: '1m', target: 40 },
    { duration: '30s', target: 15 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01'],
    errors: ['rate<0.01'],
    profile_view_duration: ['p(95)<400'],
  },
};

export function setup() {
  const registerUrl = `${BASE_URL}/api/auth/register`;
  const testUsers = [];

  for (let i = 0; i < 40; i++) {
    const userData = {
      username: `profile_user_${Date.now()}_${i}`,
      email: `profile_user_${Date.now()}_${i}@test.com`,
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

  return { users: testUsers };
}

export default function(data) {
  const user = data.users[__VU % data.users.length];
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${user.token}`
  };

  const action = Math.random();

  if (action < 0.5) {
    const profileStart = Date.now();
    const profileRes = http.get(`${BASE_URL}/api/users/${user.userId}`, { headers });
    profileViewDuration.add(Date.now() - profileStart);

    check(profileRes, {
      'profile loaded': (r) => r.status === 200,
      'profile has username': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.username === user.username;
        } catch {
          return false;
        }
      },
      'profile has stats': (r) => {
        try {
          const body = JSON.parse(r.body);
          return 'followers_count' in body && 'following_count' in body && 'tweets_count' in body;
        } catch {
          return false;
        }
      }
    }) || errorRate.add(1);

  } else if (action < 0.7) {
    const updateData = {
      display_name: `Updated Name ${Date.now()}`,
      bio: `Updated bio for ${user.username} at ${Date.now()}`
    };

    const updateRes = http.put(
      `${BASE_URL}/api/users/${user.userId}`,
      JSON.stringify(updateData),
      { headers }
    );

    const updateSuccess = check(updateRes, {
      'profile updated': (r) => r.status === 200,
      'profile has new display name': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.display_name === updateData.display_name;
        } catch {
          return false;
        }
      }
    });

    if (updateSuccess) {
      profileUpdates.add(1);
    } else {
      errorRate.add(1);
    }

  } else if (action < 0.85) {
    const followersRes = http.get(`${BASE_URL}/api/users/${user.userId}/followers`, { headers });

    check(followersRes, {
      'followers loaded': (r) => r.status === 200,
      'followers is array': (r) => {
        try {
          const body = JSON.parse(r.body);
          return Array.isArray(body);
        } catch {
          return false;
        }
      }
    }) || errorRate.add(1);

  } else {
    const followingRes = http.get(`${BASE_URL}/api/users/${user.userId}/following`, { headers });

    check(followingRes, {
      'following loaded': (r) => r.status === 200,
      'following is array': (r) => {
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
  console.log(`User profile test completed with ${data.users.length} users`);
}

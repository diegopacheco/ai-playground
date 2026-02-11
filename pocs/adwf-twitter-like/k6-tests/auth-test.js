import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const registrationDuration = new Trend('registration_duration');
const loginDuration = new Trend('login_duration');
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export const options = {
  stages: [
    { duration: '30s', target: 20 },
    { duration: '1m', target: 50 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<800'],
    http_req_failed: ['rate<0.01'],
    errors: ['rate<0.01'],
    registration_duration: ['p(95)<600'],
    login_duration: ['p(95)<400'],
  },
};

export default function() {
  const username = `auth_test_${Date.now()}_${__VU}_${Math.random().toString(36).substring(7)}`;
  const email = `${username}@test.com`;
  const password = 'password123';

  const registerData = {
    username,
    email,
    password
  };

  const registerStart = Date.now();
  const registerRes = http.post(
    `${BASE_URL}/api/auth/register`,
    JSON.stringify(registerData),
    { headers: { 'Content-Type': 'application/json' } }
  );
  registrationDuration.add(Date.now() - registerStart);

  const registerSuccess = check(registerRes, {
    'registration status 201': (r) => r.status === 201,
    'registration has token': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.token && body.token.length > 0;
      } catch {
        return false;
      }
    },
    'registration has user': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.user && body.user.username === username;
      } catch {
        return false;
      }
    }
  });

  if (!registerSuccess) {
    errorRate.add(1);
    return;
  }

  sleep(0.5);

  const loginData = {
    username,
    password
  };

  const loginStart = Date.now();
  const loginRes = http.post(
    `${BASE_URL}/api/auth/login`,
    JSON.stringify(loginData),
    { headers: { 'Content-Type': 'application/json' } }
  );
  loginDuration.add(Date.now() - loginStart);

  const loginSuccess = check(loginRes, {
    'login status 200': (r) => r.status === 200,
    'login has token': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.token && body.token.length > 0;
      } catch {
        return false;
      }
    },
    'login has user': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.user && body.user.username === username;
      } catch {
        return false;
      }
    }
  });

  if (!loginSuccess) {
    errorRate.add(1);
    return;
  }

  const loginBody = JSON.parse(loginRes.body);
  const token = loginBody.token;

  sleep(0.5);

  const logoutRes = http.post(
    `${BASE_URL}/api/auth/logout`,
    null,
    { headers: { 'Authorization': `Bearer ${token}` } }
  );

  check(logoutRes, {
    'logout status 204': (r) => r.status === 204,
  }) || errorRate.add(1);

  sleep(1);
}

export function teardown() {
  console.log('Authentication stress test completed');
}

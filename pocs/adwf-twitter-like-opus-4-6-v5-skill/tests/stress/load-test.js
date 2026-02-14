import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "10s", target: 10 },
    { duration: "20s", target: 10 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    http_req_duration: ["p(95)<500"],
    http_req_failed: ["rate<0.1"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8080/api";

export default function () {
  const uniqueId = `${__VU}_${__ITER}_${Date.now()}`;

  const registerRes = http.post(
    `${BASE_URL}/auth/register`,
    JSON.stringify({
      username: `k6user_${uniqueId}`,
      email: `k6_${uniqueId}@test.com`,
      password: "password123",
    }),
    { headers: { "Content-Type": "application/json" } }
  );
  check(registerRes, { "register status is 200": (r) => r.status === 200 });

  const loginRes = http.post(
    `${BASE_URL}/auth/login`,
    JSON.stringify({
      email: `k6_${uniqueId}@test.com`,
      password: "password123",
    }),
    { headers: { "Content-Type": "application/json" } }
  );
  check(loginRes, { "login status is 200": (r) => r.status === 200 });

  let token = "";
  if (loginRes.status === 200) {
    token = JSON.parse(loginRes.body).token;
  }

  const authHeaders = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };

  const tweetRes = http.post(
    `${BASE_URL}/tweets`,
    JSON.stringify({ content: `K6 load test tweet ${uniqueId}` }),
    { headers: authHeaders }
  );
  check(tweetRes, { "create tweet status is 201": (r) => r.status === 201 });

  const feedRes = http.get(`${BASE_URL}/tweets/feed`, {
    headers: authHeaders,
  });
  check(feedRes, { "get feed status is 200": (r) => r.status === 200 });

  sleep(0.5);
}
